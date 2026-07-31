"""The pitch-angle marginal: ``theta_smoothed``, static + movie.

The thin twin of ``vel_smoothed.py``, split into its own file so tools/run.sh
renders the two marginals as two parallel processes (the shell's unit of
parallelism is one command).  The shared :func:`derive` (which computes both marginals from one cache
pass), the drawer, and ``FIGSIZE`` are imported from ``vel_smoothed.py``;
this file only picks ``theta``.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: ``vel_smoothed.py`` for everything.
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

from plot_common.reader import load_cache, load_snapshots
from plot_common.static import render_still, save_png
from plot_common.movie import render_movie
from vel_smoothed import FIGSIZE, derive, draw_frame


def main():
    """CLI entry point: parse flags, load the data, render the figures.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    parser = argparse.ArgumentParser(description="ICRF pitch-marginal plots")
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
        draw_frame(fig, ax, data, "theta", index)

    if do_static:
        save_png(render_still(draw, len(data["times"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "theta_smoothed", dpi=220)
    if do_movie:
        render_movie(draw, len(data["times"]),
                     str(Path(args.fig_dir) / "theta_smoothed.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
