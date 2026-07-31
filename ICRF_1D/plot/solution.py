"""ICRF_1D distribution figure: f(v) versus the analytic steady state.

The 1-D isotropic speed problem has an analytic steady state, so every frame
compares the solver's state against it directly.  The companion temperature
figure lives in ``temperature.py`` (its own parallel process in tools/run.sh)
but draws entirely from this file's :func:`derive`.

``--static`` draws the final snapshot; ``--movie`` animates every snapshot
with fixed axes (default: both) -- the 1-D version of
``ICRF_2D/plot/solution.py``.

All of the collision physics needed for the analytic curves (the Chandrasekhar
G function, the multi-species eta/zeta coefficients) lives in
:class:`PlasmaData`, ported verbatim from the original ``run.py`` -- these are
the 1-D analogues of what ICRF_2D precomputes into tables, small enough here
to evaluate on the fly.

Used by: ``tools/run.sh`` (one of the parallel plotter processes), and
``temperature.py``, which imports :func:`derive` and :func:`local_temperature`.

Depends on: :mod:`plot_common.runtime` (bootstrap/paths),
:mod:`plot_common.reader` (deck + the 1-D snapshot cache),
:mod:`plot_common.static` (save_png), :mod:`plot_common.movie`
(movie rendering).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# instead of hardcoding a parent depth -- so reorganizing the project tree
# (as when everything moved into FENRIS/) cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots_1d,
    numerical_display_floor,
    option_float,
    option_vector,
    read_options,
)
from plot_common.static import render_still, save_png

# Norm of the Gaussian, 2/sqrt(pi).
GAUSS_FACT = 2.0 / np.sqrt(np.pi)

# Electron parameters after normalization: temperature and density normalize
# themselves away; the mass is in amu.
T_E = 1.0
Z_E = -1.0
M_E = 5.4461702149e-4
N_E = 1.0


def chandrasekhar_g(x, divide):
    """The Chandrasekhar function G(x) := (erf(x) - x erf'(x)) / (2 x^2).

    ``divide=True`` returns G(x)/x instead, with the x -> 0 singularity
    handled by the small-x series (G/x -> 2/(3 sqrt(pi))) so the array form
    never divides by zero.
    """
    x = np.asarray(x, dtype=float)
    derf = GAUSS_FACT * np.exp(-(x**2))
    cut = np.abs(x) < 1e-3
    small = GAUSS_FACT * (1.0 / 3.0 - x**2 / 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = (erf(x) - x * derf) / (2.0 * x**3)
    g_over_x = np.where(cut, small, big)
    return g_over_x if divide else x * g_over_x


class PlasmaData:
    """Multi-species collision coefficients and the analytic steady state.

    Species 0 is the electron (deduced from quasi-neutrality); 1..N are the
    ions from the deck.  ``eta``/``zeta`` are the drift and diffusion parts of
    the 1-D speed-space collision operator plus RF power K:

        eta(v)  := -alpha(v) + (1/2v^2) d/dv(v^2 beta(v))
        zeta(v) := beta(v)/2 + K

    and the steady state solves f'/f = -eta/zeta.
    """

    def __init__(self, K, z, m, z_ion, m_ion, n_ion, T_ion):
        """Precompute per-species coefficient arrays from deck parameters.

        ``K`` is the RF diffusion strength; ``z``/``m`` are the evolving
        minority species' charge and mass; the ``*_ion`` vectors describe the
        background ions.  Each species contributes one entry to ``l`` (its
        inverse thermal speed in normalized units), ``eta_coeff``, and
        ``beta_coeff``, so :meth:`eta` and :meth:`zeta` reduce to weighted
        sums of the Chandrasekhar function over species.
        """
        self.K = float(K)
        self.z = float(z)
        self.m = float(m)

        z_ion = np.asarray(z_ion, dtype=float)
        m_ion = np.asarray(m_ion, dtype=float)
        n_ion = np.asarray(n_ion, dtype=float)
        T_ion = np.asarray(T_ion, dtype=float)
        # Rescale ion densities so the total charge matches quasi-neutrality.
        n_ion = n_ion / np.sum(n_ion * z_ion)
        self.T_ion = T_ion

        count = n_ion.size
        self.l = np.empty(count + 1)
        self.beta_coeff = np.empty(count + 1)
        self.eta_coeff = np.empty(count + 1)

        # Entry 0: the electron.
        cf_e = N_E * Z_E**2 * self.z**2 / self.m**2
        self.l[0] = np.sqrt(0.5 * M_E / T_E)
        self.eta_coeff[0] = cf_e * self.l[0] ** 2 * self.m / M_E
        self.beta_coeff[0] = cf_e * self.l[0]

        # Entries 1..N: the ions.
        cf_i = n_ion * z_ion**2 * self.z**2 / self.m**2
        l_i = np.sqrt(0.5 * m_ion / T_ion)
        self.l[1:] = l_i
        self.eta_coeff[1:] = cf_i * l_i**2 * (self.m / m_ion)
        self.beta_coeff[1:] = cf_i * l_i

    def eta(self, v):
        """Drift coefficient: each species' G(l_s v) contribution, summed."""
        v = np.asarray(v, dtype=float)
        x = self.l[:, None] * v[None, :]
        return np.sum(self.eta_coeff[:, None] * chandrasekhar_g(x, False), axis=0)

    def zeta(self, v):
        """Diffusion coefficient: summed collisional beta/2, plus RF power K."""
        v = np.asarray(v, dtype=float)
        x = self.l[:, None] * v[None, :]
        beta = np.sum(self.beta_coeff[:, None] * chandrasekhar_g(x, True), axis=0)
        return 0.5 * beta + self.K

    def dist(self, v):
        """The normalized analytic steady state exp(-int eta/zeta dv)."""
        v = np.asarray(v, dtype=float)
        dv = v[1] - v[0]
        ratio = self.eta(v) / self.zeta(v)
        y = np.exp(-cumulative_trapezoid(ratio, v, initial=0))
        return y / (np.sum(y * v**2) * dv)

    def effective_temperature(self):
        """The flat T_eff the RF-heated steady state approaches at low v."""
        eps = GAUSS_FACT / 3.0
        drag = sum(self.beta_coeff)
        temps = np.append(1.0, self.T_ion)
        diffusion = sum(self.beta_coeff / temps)
        return (eps * drag + 2 * self.K) / (eps * diffusion)

    def large_v_temperature(self, v):
        """The large-v asymptote T ~ v^3 of the RF-dominated tail."""
        return 2 * self.m**2 / self.z**2 * M_E * self.K * v**3


def _finite_diff(values):
    """Centered difference with one-sided ends, matching the original d()."""
    delta = np.empty_like(values)
    delta[0] = values[1] - values[0]
    delta[1:-1] = values[2:] - values[:-2]
    delta[-1] = values[-1] - values[-2]
    return delta


def local_temperature(f, v, m):
    """T(v) = -f dE/df with E = m v^2 / 2, the local effective temperature."""
    energy = 0.5 * m * v**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.asarray(f, dtype=float) * _finite_diff(energy) / _finite_diff(
            np.asarray(f, dtype=float))


def derive(cache, solver_input=None):
    """Evaluate every analytic curve on the cache's grid, once for all frames.

    The analytic pieces (steady state shape, T_eff, the v^3 asymptote) do not
    depend on the frame, so they are computed here rather than per frame; each
    frame only rescales the steady curve to its own norm.
    """
    options = read_options(solver_input or PATHS.solver_input)
    v = np.asarray(cache.x, dtype=float)
    dv = v[1] - v[0]

    plasma = PlasmaData(
        K=option_float(options, "K"),
        z=option_float(options, "z"),
        m=option_float(options, "m"),
        z_ion=option_vector(options, "z_ion"),
        m_ion=option_vector(options, "m_ion"),
        n_ion=option_vector(options, "n_ion"),
        T_ion=option_vector(options, "T_ion"),
    )

    # Initial Maxwellian at the minority mass: the norm reference.
    f_initial = 2 * plasma.m**1.5 * GAUSS_FACT * np.exp(-plasma.m * v**2)
    norm_initial = np.sum(f_initial * v**2) * dv
    norms = [float(np.sum(np.asarray(f, dtype=float) * v**2) * dv)
             for f in cache.frames]

    steady_shape = plasma.dist(v)
    temp_steady = local_temperature(norms[-1] * steady_shape, v, plasma.m)

    return {
        "v": v,
        "frames": cache.frames,
        "times": cache.times,
        "plasma": plasma,
        "steady_shape": steady_shape,
        "temp_steady": temp_steady,
        "norms": norms,
        "norm_initial": norm_initial,
        "scale_dist": options.get("scale_dist", "plot"),
        "scale_temp": options.get("scale_temp", "plot"),
        # Nothing below this is trustworthy: the deck's own solver tolerance,
        # or machine error, whichever is larger.
        "floor": numerical_display_floor(
            solver_input or PATHS.solver_input),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (7.2, 5.2)


def draw_solution_frame(fig, ax, data, index):
    """One distribution frame: solved f versus the analytic steady state."""
    v, plasma = data["v"], data["plasma"]
    f = np.asarray(data["frames"][index], dtype=float)
    norm = data["norms"][index]
    f_steady = norm * data["steady_shape"]
    norm_error = abs(1 - norm / data["norm_initial"]) * 100.0

    time_label = rf"$(v/v_{{th}},\ t/\tau = {data['times'][index]:.2f})$"
    # Strict on purpose: the deck value must name a real Axes method
    # (plot, semilogy, loglog, ...); a typo should crash with the bad
    # name, not silently fall back to a linear plot.
    draw = getattr(ax, data["scale_dist"])
    draw(v, f_steady, label=r"$f$ $(v/v_{th})$ steady")
    draw(v, np.abs(f), "--", label=r"$f$ " + time_label)
    ax.set_xlabel(r"$v/v_{th}$", fontsize=14)
    ax.set_ylabel(r"$f(v/v_{th}, t/\tau)$", fontsize=14)
    # The range follows the frame: the top tracks whichever curve is larger
    # right now, and the bottom hugs the smallest value the curves actually
    # reach -- never below the numerical error floor.
    both = np.concatenate([np.abs(f), f_steady])
    positive = both[np.isfinite(both) & (both > 0.0)]
    ax.set_ylim(bottom=max(data["floor"], float(positive.min()) / 1.2),
                top=1.2 * float(positive.max()))
    ax.set_title(
        rf"$K = {plasma.K:.2f}$ $[v_{{th}}^2/\tau]$,  "
        rf"$\Delta\mathcal{{N}}\% = {norm_error:.4f}$", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle=":")
    fig.tight_layout()


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="ICRF_1D solution plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=0,
                        help="reconstruction points (0 = deck num_points / 2)")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    if args.cache:
        cache = load_cache(args.cache)
    else:
        points = args.points
        if points <= 0:
            options = read_options(PATHS.solver_input)
            points = int(option_float(options, "num_points", 256) / 2)
        cache = load_snapshots_1d(args.output, points)
    data = derive(cache)

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_solution_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["frames"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "solution", dpi=220)
    if do_movie:
        render_movie(draw, len(data["frames"]),
                     str(Path(args.fig_dir) / "solution.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
