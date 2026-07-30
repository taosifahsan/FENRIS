"""Plot everything for LHCD_2D, in one run.

The batch entry point.  Reads every snapshot exactly once, then calls into
every other plotter's ``derive``/``plot_static``/``plot_movie`` functions --
never their standalone ``main()``, so no plotter re-reads what this file
already read.

Order of work, and why:

1. **No-snapshot figures first** (``coefficients.py``) -- the diffusion
   tensor is analytic in the deck's parameters, so it needs no snapshot data
   and runs before anything has to be read.
2. **The grid lane**, started in the background -- it reads only
   ``snapshot.cells``, never a reconstruction, so it is cheap and runs
   concurrently with the expensive step below rather than after it.
3. **The one expensive step**: :func:`plot_common.reader.load_snapshots`.
   This is the only place in a batch run that reconstructs snapshots.
4. **Derive** what each plotter needs from the cache, in the parent process --
   pure numpy, no I/O, no pool.
5. **Statics**, rendered across a pool.
6. **Movies**, one at a time, each using the *full* worker budget internally
   (via :func:`plot_common.movie.render_movie`).  A single interleaved pool
   across every movie's frames would balance load slightly better still (a
   slow frame in one movie could be absorbed by another's queue), but adds
   real complexity for a difference that only shows up in the last few frames
   of whichever movie finishes first -- not worth it at this frame count.
7. Join the grid lane; archive the input decks.

Usage::

    python3 plot/show_all.py                 # everything
    python3 plot/show_all.py --static         # stills only, no movies
    python3 plot/show_all.py --movie          # movies only
    python3 plot/show_all.py -j 16 -n 256     # 16 workers, 256-point grid

Depends on: every other file in this directory, plus the shared
``plot_common`` layer.
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

import matplotlib.pyplot as plt

import coefficients
import diagnostics
import grid
import smoothed
import solution
from coefficients import style_cartesian_axes
from plot_common.movie import render_movie
from plot_common.reader import load_snapshots
from plot_common.runtime import process_pool, worker_count
from plot_common.static import cartesian_mesh, save_input_files, save_png, timestamped_output_dir


# ---------------------------------------------------------------------------
# Static figures: rendered across a pool, since each is independent
# ---------------------------------------------------------------------------


def _render_statics(jobs, output_dir, dpi, workers):
    """Run a list of ``(stem, figure_factory)`` pairs across a process pool.

    ``figure_factory`` must be zero-argument and picklable (a closure over
    plain data, not over matplotlib objects, which cannot cross a process
    boundary) -- so callers pass ``(stem, function, args)`` and this wraps
    them; see the call site.
    """
    if worker_count(workers) == 1 or len(jobs) <= 1:
        for stem, fn, fn_args in jobs:
            save_png(fn(*fn_args), output_dir, stem, dpi=dpi)
        return
    with process_pool(min(worker_count(workers), len(jobs))) as pool:
        futures = [
            pool.submit(_render_one_static, stem, fn, fn_args, output_dir, dpi)
            for stem, fn, fn_args in jobs
        ]
        for future in futures:
            future.result()


def _render_one_static(stem, fn, fn_args, output_dir, dpi):
    """Worker: build one static figure and save it.  Module-level to pickle."""
    fig = fn(*fn_args)
    save_png(fig, output_dir, stem, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Movies: one render_movie call per movie, each using the full worker budget
# ---------------------------------------------------------------------------
#
# Each entry names a plotter's per-frame draw function, the shared init data
# it needs, and how many frames it has.  render_movie is called once per
# movie; within a call, every frame is farmed across the full worker pool.


def _run_movies(specs, output_dir, workers, fps, dpi):
    for stem, frame_count, draw_task, initializer, initargs in specs:
        out = str(Path(output_dir) / f"{stem}.mp4")
        render_movie(
            draw_task, frame_count, out, fps=fps, dpi=dpi, workers=workers,
            initializer=initializer, initargs=initargs,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot everything for LHCD_2D")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-n", "--points", type=int, default=192)
    parser.add_argument("-j", "--workers", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    output_dir = timestamped_output_dir(args.fig_dir)
    workers = worker_count(args.workers)

    # --- 1. no-snapshot figures + grid lane, concurrently with nothing yet
    #        to wait for -----------------------------------------------------
    no_snapshot_jobs = [
        ("diffusion_tensor", coefficients.plot_tensor, ()),
    ]
    if do_static:
        # These have no dependency on the expensive read below, so they run
        # first and are done well before the cache finishes loading.
        _render_statics(no_snapshot_jobs, output_dir, 220, workers)

    grid_workers = max(1, workers // 8)
    grid_data = grid.load(args.output, workers=grid_workers)
    if do_static:
        save_png(grid.plot_static(grid_data), output_dir, "grid_level", dpi=220)
        save_png(grid.plot_dof(grid_data), output_dir, "grid_dof", dpi=220)
    if do_movie:
        grid.plot_movie(
            grid_data, str(Path(output_dir) / "grid_level.mp4"),
            workers=grid_workers, fps=args.fps, dpi=args.dpi,
        )

    # --- 2. the one expensive step ------------------------------------------
    solution_workers = max(1, workers - grid_workers)
    cache = load_snapshots(args.output, args.points, workers=solution_workers)

    # --- 3. derive everything in the parent: pure numpy, no I/O -------------
    solution_data = solution.derive(cache)
    smoothed_data = smoothed.derive(cache)
    diagnostics_data = diagnostics.derive(cache)
    vpar, vperp = cartesian_mesh(cache.x, cache.y)

    # --- 4. statics from the cache ------------------------------------------
    if do_static:
        save_png(solution.plot_static(solution_data), output_dir, "solution",
                 dpi=220)
        for reduction in smoothed.REDUCTIONS:
            save_png(smoothed.plot_static(smoothed_data, reduction), output_dir,
                     f"{reduction}_smoothed", dpi=220)
        save_png(
            diagnostics.plot_growth_static(
                vpar, vperp, style_cartesian_axes, diagnostics_data["growth"]
            ),
            output_dir, "growth_rate", dpi=220,
        )
        save_png(diagnostics.plot_particle_loss(diagnostics_data["conservation"]),
                 output_dir, "particle_loss", dpi=220)
        save_png(diagnostics.plot_energy_power(diagnostics_data["conservation"]),
                 output_dir, "energy_power", dpi=220)

    # --- 5. movies -----------------------------------------------------------
    if do_movie:
        _run_movies(
            [("solution", len(solution_data["frames"]),
              solution._draw_solution_frame_task, solution._init_solution_worker,
              (solution_data,))],
            output_dir, solution_workers, args.fps, args.dpi,
        )
        for reduction in smoothed.REDUCTIONS:
            _run_movies(
                [(f"{reduction}_smoothed", len(smoothed_data["times"]),
                  smoothed._draw_smoothed_frame_task,
                  smoothed._init_smoothed_worker, (smoothed_data, reduction))],
                output_dir, solution_workers, args.fps, args.dpi,
            )
        _run_movies(
            [("growth_rate", len(diagnostics_data["growth"]["frames"]),
              diagnostics._draw_growth_frame_task,
              diagnostics._init_growth_worker,
              (vpar, vperp, style_cartesian_axes, diagnostics_data["growth"]))],
            output_dir, solution_workers, args.fps, args.dpi,
        )

    save_input_files(output_dir, PATHS.solver_input)
    print(f"saved everything in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
