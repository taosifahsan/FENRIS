"""The solution: the 2-D distribution function ``F_0(v_parallel, v_perp)``.

Cartesian view, signed-log color -- see ``plot_common/static.py`` for why both
are fixed rather than flag-selectable.

``--static`` draws the final snapshot; ``--movie`` animates every snapshot
with one fixed color scale (so a color means the same value in every frame).
Both read through :func:`plot_common.reader.load_snapshots` exactly once --
this is the plotter most other plots in this project piggyback on when run
from ``tools/run.sh``, since the solution reconstruction is the expensive step
every other 2-D/marginal figure also needs.

Used by:
  - ``tools/run.sh`` -- runs this as one of the parallel plotter processes

Depends on: :mod:`plot_common.reader` (the cache), :mod:`plot_common.static`
(drawing), :mod:`plot_common.movie` (parallel frame rendering),
``coefficients.py`` (the initial-condition overlay's normalization).
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


def derive(cache, solver_input=None, table_dir=None):
    """Turn a raw snapshot cache into what the solution plots need.

    Computes the Cartesian mesh once (coordinates are the same for every
    frame), the initial-condition overlay on that mesh, and one movie-wide
    color scale so every frame -- static or animated -- uses the same
    colorbar.
    """
    vpar, vperp = cartesian_mesh(cache.x, cache.y)
    initial = initial_condition_grid(cache.x, cache.y, solver_input, table_dir)
    scale = movie_scale_range(cache.frames)
    floor = numerical_display_floor(solver_input or PATHS.solver_input)
    return {
        "frames": cache.frames,
        "times": cache.times,
        "vpar": vpar,
        "vperp": vperp,
        "initial": initial,
        "scale": scale,
        "floor": floor,
    }


def _title(time):
    return rf"$\mathcal{{F}}_0$:  time, $t = {time:.2f}\,\tau_c$"


# One size for the still and every movie frame -- stated once.
FIGSIZE = (5.4, 4.5)


def draw_frame(fig, ax, data, index):
    """Draw one solution frame from already-derived data."""
    contour2d(
        fig, ax, data["vpar"], data["vperp"], data["frames"][index],
        fixed_range=data["scale"], title=_title(data["times"][index]),
        style_axes=style_cartesian_axes, floor=data["floor"],
        restrict_levels=True,
    )
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.84)


def main():
    parser = argparse.ArgumentParser(description="ICRF solution plot")
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
                 args.fig_dir, "solution", dpi=220)
    if do_movie:
        render_movie(draw, len(data["frames"]),
                     str(Path(args.fig_dir) / "solution.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
