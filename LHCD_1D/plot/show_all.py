"""Render every LHCD_1D figure and movie from one snapshot pass.

The 1-D version of ``LHCD_2D/plot/show_all.py``: everything lands in a fresh
timestamped directory under ``figures/``, with the input deck copied alongside
so the run is self-describing months later.

Snapshots are read exactly once into the shared 1-D cache and handed to the
solution/temperature and growth plotters; ``grid.py`` does its own far cheaper
cells-only read.

Outputs: solution, grid_level, grid_dof, growth_average (PNG);
solution, grid_level, growth_rate (MP4).

Used by: the ``plots`` CMake target (and by hand).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# instead of hardcoding a parent depth -- so reorganizing the project tree
# cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

import diagnostics
import grid
import solution
from plot_common.reader import load_snapshots_1d, option_float, read_options
from plot_common.static import save_input_files, save_png, timestamped_output_dir


def main():
    parser = argparse.ArgumentParser(description="LHCD_1D: all figures + movies")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=0,
                        help="reconstruction points (0 = deck num_points / 2)")
    parser.add_argument("-j", "--workers", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    out = timestamped_output_dir(args.fig_dir)

    points = args.points
    if points <= 0:
        points = int(option_float(read_options(PATHS.solver_input),
                                  "num_points", 256) / 2)

    # Grid lane first: cells-only read, no reconstruction.
    grid_data = grid.load(args.output, workers=args.workers)
    if do_static:
        save_png(grid.plot_static(grid_data), out, "grid_level", dpi=220)
        save_png(grid.plot_dof(grid_data), out, "grid_dof", dpi=220)
    if do_movie:
        grid.plot_movie(grid_data, str(Path(out) / "grid_level.mp4"),
                        workers=args.workers, fps=args.fps, dpi=args.dpi)

    # The one expensive read, shared by every remaining plotter.
    cache = load_snapshots_1d(args.output, points, workers=args.workers)
    solution_data = solution.derive(cache)
    growth_data = diagnostics.derive(cache)

    if do_static:
        save_png(solution.plot_static(solution_data), out, "solution", dpi=220)
        save_png(diagnostics.plot_average(growth_data), out,
                 "growth_average", dpi=220)
        save_png(diagnostics.plot_particles(growth_data), out,
                 "particle_loss", dpi=220)
        save_png(diagnostics.plot_energy_power(growth_data), out,
                 "energy_power", dpi=220)
    if do_movie:
        solution.plot_movie(
            solution_data, str(Path(out) / "solution.mp4"),
            workers=args.workers, fps=args.fps, dpi=args.dpi)
        diagnostics.plot_movie(
            growth_data, str(Path(out) / "growth_rate.mp4"),
            workers=args.workers, fps=args.fps, dpi=args.dpi)

    save_input_files(out, PATHS.solver_input)
    print(f"saved everything in {out}", flush=True)


if __name__ == "__main__":
    main()
