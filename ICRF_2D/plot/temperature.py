"""ICRF_2D local effective temperature of the speed marginal: static + movie.

The 2-D cousin of ``ICRF_1D/plot/temperature.py``, built on the vel-smoothed
marginal instead of the full distribution:

    T(x_0) = -F_0 dE/dF_0,   E = x_0^2 / 2   (normalized units)

For a Maxwellian ``exp(-x_0^2/2)`` this is exactly 1, so the initial frame is
a constant line; RF heating then lifts the tail as the movie progresses.
The initial line is the constant ``T_bg = 1`` (exact while every deck
species shares one temperature -- see the note in :func:`derive`); each
frame's curve is finite-differenced from the reconstructed marginal.  No
steady-state overlays (unlike the 1-D version): just initial versus
current, in the same layout as ``vel_smoothed``.

Wherever |F_0| is below the numerical display floor the temperature is a
ratio of noise over noise and is masked out (drawn as a gap).

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: ``vel_smoothed.py`` (the shared :func:`derive`, the marginal
itself, and ``FIGSIZE``), :mod:`plot_common` for everything mechanical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# so reorganizing the project tree cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots
from plot_common.static import render_still, save_png
from vel_smoothed import FIGSIZE, derive as derive_marginal


def _finite_diff(values):
    """Centered difference with one-sided ends (same scheme as ICRF_1D)."""
    delta = np.empty_like(values)
    delta[0] = values[1] - values[0]
    delta[1:-1] = values[2:] - values[:-2]
    delta[-1] = values[-1] - values[-2]
    return delta


def local_temperature(f, x, floor):
    """T(x_0) = -dE/d(ln f) with E = x_0^2, in units of the background T.

    ``E = x_0^2`` because the code's speed variable is ``x_0 = v/sqrt(2T/m)``
    (the initial equilibrium is exactly ``exp(-x_0^2)``, measured slope
    -1.0000), so this normalization makes the initial temperature read 1.

    Differencing ``ln f`` rather than ``f`` makes T *exact* for pure
    exponentials -- a plain ``-f dE/df`` finite difference under-reads a
    fast-decaying Maxwellian by ~(x dx)^2, a visible droop by x_0 ~ 4.

    Below the display floor f is solver noise, so d(ln f) is noise and the
    ratio is meaningless -- masked to NaN, drawn as gaps.
    """
    f = np.asarray(f, dtype=float)
    energy = np.asarray(x, dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        temp = -_finite_diff(energy) / _finite_diff(np.log(np.abs(f)))
    temp[~np.isfinite(temp)] = np.nan
    temp[np.abs(f) < floor] = np.nan
    return temp


def derive(cache, solver_input=None):
    """Temperature of every frame's speed marginal, plus a fixed y-range.

    Rides entirely on :func:`vel_smoothed.derive`: one marginal reduction of
    the shared cache, then one finite-difference pass per frame.
    """
    base = derive_marginal(cache, solver_input=solver_input)
    branch = base["vel"]
    x = np.asarray(branch["x"], dtype=float)
    floor = base["floor"]

    # The initial (pre-RF) temperature is T_bg = 1 by construction: every
    # background species in the deck sits at the same temperature, so the
    # collisional equilibrium is exp(-x^2) exactly.  NOTE: a deck with
    # unequal species temperatures makes this line wrong -- the general
    # curve is 2x B/A via coefficients.drag_over_diffusion (harmonic mixture
    # of species temperatures), one import away.
    initial = np.ones_like(x)

    frames = [local_temperature(f, x, floor) for f in branch["frames"]]

    # Fixed movie-wide y-range: from half the initial (constant) temperature
    # up to 1.2x the hottest value any frame reaches.
    t0 = float(np.nanmedian(initial))
    top = max(float(np.nanmax(t)) for t in frames if np.isfinite(t).any())
    return {
        "x": x,
        "initial": initial,
        "frames": frames,
        "times": base["times"],
        "ylim": (0.5 * t0, 1.2 * max(top, t0)),
    }


def draw_frame(fig, ax, data, index):
    """One temperature frame: initial (dashed) versus current (solid)."""
    ax.semilogy(data["x"], data["initial"], color="#171717", linestyle="--",
                lw=1.8, label=r"initial ($T_{bg}$)")
    ax.semilogy(data["x"], data["frames"][index], color="#118ab2",
                linestyle="-", lw=2.0, label="current")
    ax.set_ylim(*data["ylim"])
    ax.set_xlabel(r"$x_0$", fontsize=13)
    ax.set_ylabel(r"$T(x_0)\,/\,T_{bg}$", fontsize=13)
    time_label = rf"time, $t = {data['times'][index]:.2f}\,\tau_c$"
    ax.set_title(r"$T(x_0)=-\mathcal{F}_0\,dE/d\mathcal{F}_0$"
                 f":  {time_label}", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="ICRF_2D temperature plots")
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
    data = derive(cache)

    def draw(fig, ax, index):
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["frames"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "temperature", dpi=220)
    if do_movie:
        render_movie(draw, len(data["frames"]),
                     str(Path(args.fig_dir) / "temperature.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
