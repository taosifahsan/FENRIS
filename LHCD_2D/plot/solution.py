"""The solution: the 2-D distribution function ``f(v_parallel, v_perp)``.

Cartesian view, signed-log color.  ``--static`` draws the final snapshot;
``--movie`` animates every snapshot with one fixed color scale, so a color
means the same value in every frame.

Structurally identical to ``ICRF_2D/plot/solution.py``; the only difference is
the initial-condition overlay (LHCD's is an analytic Maxwellian, ICRF's a
table-derived collisional equilibrium) and the axis labels.

Used by: ``tools/run.sh`` -- runs this as a parallel plotter process on the shared
cache.

Depends on: :mod:`plot_common.reader` (the cache), :mod:`plot_common.static`
(drawing), :mod:`plot_common.movie` (movie rendering),
``coefficients.py`` (the initial condition and axis styling).
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

from coefficients import initial_condition_grid, style_cartesian_axes
from plot_common.movie import render_movie
from plot_common.reader import load_cache, load_snapshots, numerical_display_floor
from plot_common.static import (
    render_still,
    cartesian_mesh,
    contour2d,
    movie_scale_range,
    save_png,
)


def derive(cache, solver_input=None):
    """Turn a raw snapshot cache into what the solution plots need.

    Computes the Cartesian mesh once (coordinates are identical for every
    frame), the initial-condition overlay on that mesh, and one movie-wide
    color scale so every frame -- static or animated -- shares a colorbar.
    """
    vpar, vperp = cartesian_mesh(cache.x, cache.y)
    return {
        "frames": cache.frames,
        "times": cache.times,
        "vpar": vpar,
        "vperp": vperp,
        "initial": initial_condition_grid(cache.x),
        "scale": movie_scale_range(cache.frames),
        "floor": numerical_display_floor(solver_input or PATHS.solver_input),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.4, 4.5)


def draw_frame(fig, ax, data, index):
    """Draw one solution frame from already-derived data."""
    contour2d(
        fig, ax, data["vpar"], data["vperp"], data["frames"][index],
        fixed_range=data["scale"],
        title=rf"$f$:  time, $t = {data['times'][index]:.2f}\,\tau_c$",
        style_axes=style_cartesian_axes, floor=data["floor"],
        restrict_levels=True,
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="LHCD solution plot")
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
