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

Used by: the ``plots`` CMake target (and by hand).

Depends on: :mod:`plot_common.runtime` (bootstrap/paths),
:mod:`plot_common.reader` (deck + the 1-D snapshot cache),
:mod:`plot_common.static` (save_png), :mod:`plot_common.movie`
(parallel frame rendering).
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
import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import (
    load_snapshots_1d,
    numerical_display_floor,
    option_float,
    read_options,
)
from plot_common.static import save_png


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
    draw = getattr(ax, scale, ax.plot)
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


def plot_static(data):
    """Final-snapshot figure."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_frame(fig, ax, data, len(data["frames"]) - 1)
    return fig


# Per-worker state for the movie, sent once via the pool initializer.
_DATA = None


def _init_solution_worker(data):
    global _DATA
    _DATA = data


def _draw_solution_frame_task(task):
    """Worker: draw and save one frame.

    No ``bbox_inches="tight"``: fixed ``figsize x dpi`` keeps every frame's
    pixel dimensions identical (and even), which H.264's ``yuv420p`` requires.
    """
    index = task["index"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_frame(fig, ax, _DATA, index)
    fig.savefig(f"{task['frame_dir']}/frame_{index:06d}.png", dpi=task["dpi"])
    plt.close(fig)


def plot_movie(data, output_file, *, workers=None, fps=8, dpi=140):
    """Render the solution movie, one frame per snapshot."""
    return render_movie(
        _draw_solution_frame_task, len(data["frames"]), output_file,
        fps=fps, dpi=dpi, workers=workers,
        initializer=_init_solution_worker, initargs=(data,),
    )


def main():
    parser = argparse.ArgumentParser(description="LHCD_1D solution plots")
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

    points = args.points
    if points <= 0:
        options = read_options(PATHS.solver_input)
        points = int(option_float(options, "num_points", 256) / 2)

    cache = load_snapshots_1d(args.output, points, workers=args.workers)
    data = derive(cache)

    if do_static:
        save_png(plot_static(data), args.fig_dir, "solution", dpi=220)
    if do_movie:
        plot_movie(data, str(Path(args.fig_dir) / "solution.mp4"),
                   workers=args.workers, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
