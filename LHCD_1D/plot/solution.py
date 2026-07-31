"""LHCD_1D's one figure: the parallel-velocity distribution f(v_parallel).

``--static`` draws the final snapshot; ``--movie`` animates every snapshot
with fixed axes (default: both) -- the 1-D version of
``LHCD_2D/plot/solution.py``.

The deck chooses presentation, exactly as the original ``run.py`` did:

  plot_scale : plot | semilogy | loglog | semilogx
      semilogy splits into f>0 / |f<0| branches (a log axis cannot show
      negatives); loglog/semilogx mirror the domain onto |v_parallel| and
      show the v>0 and v<0 halves as two curves.
  normalize  : yes | no  -- divide by the current norm before drawing.

Each frame's title reports the running mean and standard deviation of the
distribution, so the movie shows the quasilinear plateau forming numerically
as well as visually.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

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

from plot_common.movie import render_movie
from plot_common.reader import (
    load_cache,
    load_snapshots_1d,
    numerical_display_floor,
    option_float,
    read_options,
)
from plot_common.static import render_still, save_png


def distribution_stats(v, f):
    """Norm, mean, and standard deviation of one frame, by trapezoid sums."""
    f = np.asarray(f, dtype=float)
    dv = v[1] - v[0]
    norm = np.sum(f) * dv
    mean = np.sum(v * f) * dv / norm
    variance = np.sum((v - mean) ** 2 * f) * dv / norm
    return norm, mean, float(np.sqrt(variance))


def derive(cache, solver_input=None):
    """Per-frame stats plus the presentation choices, computed once."""
    options = read_options(solver_input or PATHS.solver_input)
    v = np.asarray(cache.x, dtype=float)
    floor = numerical_display_floor(solver_input or PATHS.solver_input)

    stats = [distribution_stats(v, frame) for frame in cache.frames]
    normalize = options.get("normalize", "yes") == "yes"

    # Fixed y-limit spanning every (possibly normalized) frame, so movie axes
    # do not rescale as the plateau forms.
    peak = max(
        float(np.nanmax(np.abs(frame))) / (norm if normalize else 1.0)
        for frame, (norm, _, _) in zip(cache.frames, stats)
    )

    return {
        "v": v,
        "frames": cache.frames,
        "times": cache.times,
        "stats": stats,
        "normalize": normalize,
        "scale": options.get("plot_scale", "plot"),
        "peak": peak,
        # Below this the values are adaptive-grid (or machine) error, not
        # solution: the display cuts there instead of plotting noise.
        "floor": floor,
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (7.2, 5.0)


def draw_frame(fig, ax, data, index):
    """Draw one distribution frame in the deck's chosen scale."""
    v = data["v"]
    f = np.asarray(data["frames"][index], dtype=float)
    norm, mean, std = data["stats"][index]
    if data["normalize"]:
        f = f / norm
        ax.set_ylabel(r"$f(v_{\parallel})\,/\int dv\, f(v)$")
    else:
        ax.set_ylabel(r"$f(v_{\parallel})$")

    scale = data["scale"]
    # Strict on purpose: a deck typo crashes with the bad name rather
    # than silently falling back to a linear plot.
    draw = getattr(ax, scale)
    if scale in ("loglog", "semilogx"):
        # A log x-axis cannot show v<0, so mirror onto |v_parallel| and show
        # the two halves as separate curves.
        draw(v, f, label=r"$v_{\parallel}>0$")
        draw(-v, f, "--", label=r"$v_{\parallel}<0$")
        ax.set_xlabel(r"parallel velocity, $|v_{\parallel}|$")
        ax.legend(loc="best")
    elif scale == "semilogy" and float(np.nanmin(f)) < 0:
        # A log y-axis cannot show f<0: split into sign branches.
        draw(v, f, label=r"$f(v_{\parallel}) > 0$")
        draw(v, -f, label=r"$f(v_{\parallel}) < 0$")
        ax.set_xlabel(r"parallel velocity, $v_{\parallel}$")
        ax.legend(loc="best")
    else:
        draw(v, f)
        ax.set_xlabel(r"parallel velocity, $v_{\parallel}$")

    bottom = data["floor"] / (norm if data["normalize"] else 1.0)
    if scale in ("semilogy", "loglog"):
        ax.set_ylim(bottom=bottom, top=1.2 * data["peak"])
    else:
        ax.set_ylim(top=1.2 * data["peak"])
    ax.set_title(
        rf"$t = {data['times'][index]:.2f}$:  "
        rf"$\langle v_{{\parallel}} \rangle = {mean:.2f}$,  "
        rf"$\sigma_{{v_{{\parallel}}}} = {std:.2f}$"
    )
    ax.grid(True, which="major", linestyle=":")
    fig.tight_layout()


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="LHCD_1D solution plots")
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
        draw_frame(fig, ax, data, index)

    if do_static:
        save_png(render_still(draw, len(data["frames"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "solution", dpi=220)
    if do_movie:
        render_movie(draw, len(data["frames"]),
                     str(Path(args.fig_dir) / "solution.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
