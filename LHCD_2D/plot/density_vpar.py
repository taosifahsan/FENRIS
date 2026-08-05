"""The parallel-velocity density ``n(v_par) = 2 pi * integral f v_perp dv_perp``.

The marginal the resonance actually lives on: LHCD's quasilinear window is
a band in ``x_parallel = x cos(theta)``, so this is the axis where the
Fisch plateau appears with sharp edges (in the speed marginal ``n(x)`` the
window smears over ``x >= x_par``).

The weight is exact, not modeled: changing velocity coordinates from the
solver's spherical chart to cylindrical ``(v_par, v_perp, phi)`` turns the
volume element ``x^2 sin(theta) dx dtheta dphi`` into
``v_perp dv_par dv_perp dphi`` (the metric determinant transforms by the
Jacobian ``dv_par dv_perp = x dx dtheta``), and the gyrophase integral
gives the ``2 pi``.  So ``integral n dv_par`` is the full 3-D particle
count -- equal to 1 at t=0 by the initial-condition normalization, and to
``2 pi *`` ``diagnostics.py``'s number moment at every frame, which is the
consistency check to reach for if this reduction is ever in doubt.  Unlike
a native-axis marginal, the fixed-``v_par`` integration lines cut
diagonally across the reconstruction grid, so each frame is bilinearly
interpolated along them rather than reduced by a per-axis weight vector.

**The analytic background** is the final steady state of the reduced 1-D
equation derived in ``Readings/2D-1D-Fokker-Planck-Derivation.pdf``:
integrating the full 2-D equation over ``pi dv_perp^2`` with the
Maxwellian-perpendicular ansatz and the ``w^2 >> x^2`` tail approximation.
NOTE ON UNITS: the document works in the old ``v_th = sqrt(T/m)``
convention (equilibrium ``e^{-w^2/2}``, ``D_c = 1/(2w^3)``); this project
standardized to ``v_th = sqrt(2T/m)`` (equilibrium ``e^{-w^2}``,
``D_c = 1/(4w^3)``, ``A = 1/(2w^2)`` -- see src/LHCD_2D.cpp's normalization
header).  The reduction's structure survives unchanged -- both collisional
pieces multiplied by (Zi+2) -- and its steady state in the standard units is

    F'/F = -[(Zi+2)/(2 w^2)] / [ D(w) + (Zi+2)/(4 w^3) ]

(the document's formula with its w equal to sqrt(2) times ours; only the
diffusion denominator's coefficient changes, the drag term is literally
identical).

Both collision channels survive the reduction as parallel drag+diffusion:
the speed channel contributes 1x the naive ``w``-evaluated coefficients and
pitch-angle scattering contributes ``(Zi+1)x`` more -- the ``(Zi+2)``
factor -- so pitch scattering is the DOMINANT parallel collisionality in
the marginal (at Zi=1, two thirds of it).  ``Zi`` and the window come from
this deck; ``eps ~ 0.01`` regularizes ``w -> 0`` for display (the
derivation's ``w >> x`` regime does not reach the bulk anyway, where the
curve reduces to the exact Maxwellian).  Normalized so its total particle
count equals the initial frame's, and drawn as a fixed background in the
still and every movie frame.

**How well it actually does**, measured against a converged run with the
pitch operator switched off (``Zi = -1.0``, so ``(Zi+1)/(4x^3) = 0`` and the
Maxwellian-``v_perp`` closure is very nearly exact -- see
``density_vperp.py``).  The window fill ``<n/this curve>`` approaches
**0.91** with a time constant of ~45 tau_c (0.84 at t=100; the asymptote is
a saturating-exponential fit, stable at 0.905-0.916 over fit windows).  So
the derivation is right about the SHAPE and slightly high on the LEVEL: the
ratio is flat across the window to within 1%, i.e. a pure amplitude offset
with no slope error at all.  That ~9% is the size of the derivation's own
``w^2 >> x^2`` tail approximation, whose leading correction is
``<x_perp^2>/w^2`` = 16%/12%/9% at w = 3.5/4.0/4.7 -- not a closure failure.

With pitch scattering ON (``Zi = 1.0``) the comparison inverts and the fill
blows THROUGH 1.0 (past it near t=39, 1.11 by t=50, still climbing).  Mind
the sign of that feedback: every collision coefficient falls like
``1/x^3``, so the fat ``v_perp`` tail that pitch scattering builds means
larger ``x`` at fixed ``v_par`` and therefore WEAKER parallel
collisionality, so the true plateau sits ABOVE this curve, not below.
``density_vperp.py`` documents why that case has no steady state at all.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.reader` (the cache and the deck),
:mod:`plot_common.static` (drawing), :mod:`plot_common.movie` (movie
rendering), scipy (bilinear frame interpolation).
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
from scipy.interpolate import RegularGridInterpolator

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots,
    numerical_display_floor,
    option_float,
    read_options,
)
from plot_common.static import line1d, render_still, save_png


def _oriented_axes(x0, theta0):
    """Sorted 1-D ``(x, theta)`` axes plus the reorientation for each frame.

    Empirical axis detection (same check the other 2-D plotters use):
    whichever axis ``theta0`` varies along is the pitch axis; guessing wrong
    would interpolate the transposed field while still looking plausible.
    """
    change0 = np.nanmax(np.abs(np.diff(theta0, axis=0))) if theta0.shape[0] > 1 else 0.0
    change1 = np.nanmax(np.abs(np.diff(theta0, axis=1))) if theta0.shape[1] > 1 else 0.0
    pitch_axis = 0 if change0 > change1 else 1
    x = np.take(x0, 0, axis=pitch_axis)
    theta = np.take(theta0, 0, axis=1 - pitch_axis)
    x_order, theta_order = np.argsort(x), np.argsort(theta)

    def orient(frame):
        values = frame if pitch_axis == 1 else frame.T
        return values[np.ix_(x_order, theta_order)]

    return x[x_order], theta[theta_order], orient


def analytic_steady(vpar, solver_input=None):
    """The reduced 1-D equation's steady state (un-normalized shape).

    See the module header: the (Zi+2)/2 coefficients from the 2D->1D
    derivation, deck window, ln F integrated by cumulative trapezoid on a
    fine grid.  The window never opens for v < 0
    (``cut_x_parallel_min > 0``), so the negative side is the pure
    Maxwellian branch.
    """
    options = read_options(solver_input or PATHS.solver_input)
    cut_min = option_float(options, "cut_x_parallel_min")
    cut_max = option_float(options, "cut_x_parallel_max")
    height = option_float(options, "cut_height")
    width = float(options.get("smoothing_width", "0.02"))
    eps = float(options.get("epsilon", "0.01"))
    zi = option_float(options, "Zi")

    def logistic(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))

    def slope(w):
        """-d ln F/dw at signed w, in the standard sqrt(2T/m) units.

        Effective drag over (window + effective diffusion), with the
        reduction's (Zi+2) multiplying both collisional pieces: the speed
        channel's D_c = 1/(4w^3), A = 1/(2w^2) survive at 1x and pitch
        scattering adds (Zi+1)x more.  No-RF check:
        drag/diffusion -> 2|w| = -d ln exp(-w^2)/dw, the solver's Maxwellian.
        """
        window = (height * logistic((w - cut_min) / width)
                  * logistic((cut_max - w) / width)) if width > 0 else (
            height * ((cut_min < w) & (w < cut_max)))
        drag = 0.5 * (zi + 2.0) / (w * w + eps * eps)
        diffusion = window + 0.25 * (zi + 2.0) * (w * w + eps * eps) ** -1.5
        return drag / diffusion

    # ln f(v) - ln f(0) = -integral_0^|v| slope(+-s) ds: decreasing away
    # from v = 0 on both sides; only the (one-sided) window breaks the
    # symmetry.
    fine = np.linspace(0.0, float(np.max(np.abs(vpar))), 4001)
    up = slope(fine)
    log_pos = -np.concatenate([[0.0], np.cumsum(0.5 * (up[1:] + up[:-1]) * np.diff(fine))])
    down = slope(-fine)
    log_neg = -np.concatenate([[0.0], np.cumsum(0.5 * (down[1:] + down[:-1]) * np.diff(fine))])

    vpar = np.asarray(vpar, dtype=float)
    return np.exp(np.where(vpar >= 0.0,
                           np.interp(np.abs(vpar), fine, log_pos),
                           np.interp(np.abs(vpar), fine, log_neg)))


def derive_density(cache, solver_input=None):
    """Reduce every frame to n(v_par) and build the normalized overlay.

    The fixed-``v_par`` lines are diagonals of the ``(x, theta)`` grid, so
    the reduction interpolates each frame bilinearly at
    ``x = hypot(v_par, v_perp)``, ``theta = atan2(v_perp, v_par)`` and
    integrates ``2 pi f v_perp`` over ``v_perp`` by trapezoid.  The query
    points depend only on the fixed grid, so they are built once and reused
    for every frame.
    """
    x, theta, orient = _oriented_axes(cache.x, cache.y)
    x_max = float(x[-1])
    floor = numerical_display_floor(solver_input or PATHS.solver_input)

    vpar = np.linspace(-x_max, x_max, 401)
    # One v_perp grid per vpar, spanning the disk |v| <= x_max.
    span = np.sqrt(np.maximum(x_max**2 - vpar**2, 0.0))
    vperp = np.linspace(0.0, 1.0, 400)[np.newaxis, :] * span[:, np.newaxis]
    xq = np.clip(np.hypot(vpar[:, np.newaxis], vperp), x[0], x[-1])
    thq = np.clip(np.arctan2(vperp, vpar[:, np.newaxis]), theta[0], theta[-1])
    query = np.stack([xq.ravel(), thq.ravel()], axis=-1)

    def reduce(frame):
        values = RegularGridInterpolator((x, theta), orient(np.asarray(frame, float)),
                                         bounds_error=False, fill_value=0.0)(query)
        integrand = values.reshape(vperp.shape) * vperp
        return 2.0 * np.pi * np.trapezoid(integrand, x=vperp, axis=1)

    frames = [reduce(frame) for frame in cache.frames]
    numbers = [float(np.trapezoid(frame, vpar)) for frame in frames]

    # The analytic background, normalized to the INITIAL total so its area
    # is the particle count the run started with.
    shape = analytic_steady(vpar, solver_input)
    analytic = shape * (numbers[0] / float(np.trapezoid(shape, vpar)))

    # Signed-log y-limits spanning the analytic curve and every frame, same
    # convention as the other density plots.
    everything = np.concatenate([np.abs(analytic)] + [np.abs(fr) for fr in frames])
    finite = everything[np.isfinite(everything) & (everything > 0.0)]
    return {
        "vpar": vpar,
        "analytic": analytic,
        "frames": frames,
        "numbers": numbers,
        "times": cache.times,
        "ylim": (floor, float(finite.max()) * 1.1),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.9, 3.8)


def draw_frame(fig, ax, data, index):
    """One frame: the analytic steady background (dashed) and current n."""
    line1d(
        ax, data["vpar"],
        [
            (data["analytic"], "1-D reduced steady state", {"color": "#999999",
                                                            "linestyle": "--", "lw": 1.8}),
            (data["frames"][index], "current", {"color": "#118ab2",
                                                "linestyle": "-", "lw": 2.0}),
        ],
        scale="log", ylim=data["ylim"], legend=False,
    )
    ax.set_xlabel(r"$v_\parallel/v_{te}$", fontsize=13)
    ax.set_ylabel(r"$n(v_\parallel)$", fontsize=13)
    ax.set_title(
        r"$n(v_\parallel)=2\pi\int f\,v_\perp\,dv_\perp$"
        rf":  $t={data['times'][index]:.2f}\,\tau_c$,"
        rf"  $N={data['numbers'][index]:.4f}$",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="LHCD parallel-velocity density plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--cache", default=None,
                        help="load the shared cache.npz instead of reading snapshots")
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=192)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    if args.cache:
        cache = load_cache(args.cache)
    else:
        cache = load_snapshots(args.output, args.points)
    data = derive_density(cache)

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["times"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "density_vpar", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "density_vpar.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
