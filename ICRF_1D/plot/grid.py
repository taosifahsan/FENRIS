"""The adaptive sparse grid, one dimension: refinement level and DOF count.

The 1-D version of ``ICRF_2D/plot/grid.py``:

1. **Refinement-level profile** (static: final frame; movie: one frame per
   snapshot) -- a step curve of "how many times has the grid been refined
   here" versus velocity.  In 1-D the 2-D filled contour collapses to a line.
2. **Degrees-of-freedom history** -- active cell count versus time.

Reads only ``snapshot.cells`` (the active hierarchical indices), never a
reconstruction, so it is far cheaper than the solution pipeline.

Used by: the ``plots`` CMake target (and by hand).

Depends on: :mod:`plot_common.reader` (``read_adaptive_grid``, deck),
:mod:`plot_common.static` (``line1d``, ``save_png``),
:mod:`plot_common.movie` (parallel frame rendering).
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

import matplotlib.pyplot as plt
import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import (
    option_number,
    read_adaptive_grid,
    read_options,
    snapshot_files,
)
from plot_common.static import line1d, render_still, save_png

# See ICRF_2D/plot/grid.py for the full explanation of the hierarchical
# numbering: 1-D index p sits at level bit_length(p)-1; indices 0 and 1 are
# both the root cell covering the whole domain.

SAMPLES_PER_LEAF = 4


def hierarchical_interval(index, domain_left, domain_right):
    """Physical ``(left, right)`` span of one hierarchical cell."""
    if index <= 1:
        return domain_left, domain_right
    scale = 1 << (int(index).bit_length() - 1)
    unit_left = index / scale - 1.0
    unit_right = (index + 1) / scale - 1.0
    width = domain_right - domain_left
    return domain_left + width * unit_left, domain_left + width * unit_right


def _resolution_for_max_level(max_level, cap=32769):
    """Sample count matched to the run's finest possible cell (1-D is cheap,
    so the cap is generous)."""
    return min(cap, SAMPLES_PER_LEAF * (1 << max(0, int(max_level))) + 1)


def refinement_levels(cells, domain_min, domain_max, points):
    """Deepest active refinement level covering each sample point.

    Every active cell paints its own interval with its own level; overlaps
    (ancestors) resolve by maximum, exactly as in the 2-D version.
    Returns ``(v, level)``.
    """
    v = np.linspace(float(domain_min[0]), float(domain_max[0]), points)
    level = np.zeros(points, dtype=int)
    for index in np.unique(np.asarray(cells, dtype=int).ravel()):
        index = int(index)
        left, right = hierarchical_interval(index, v[0], v[-1])
        first = int(np.searchsorted(v, left, side="left"))
        last = int(np.searchsorted(v, right, side="right"))
        if first >= last:
            continue
        depth = max(0, index.bit_length() - 1)
        np.maximum(level[first:last], depth, out=level[first:last])
    return v, level


def load(snapshot_dir, solver_input=None):
    """Read the grid structure of every snapshot (cheap, serial).

    Returns records (time-sorted ``(cells, dmin, dmax, time, num_cells)``),
    ``times``, ``dofs``, and the deck-derived sampling ``resolution``.
    """
    options = read_options(solver_input or PATHS.solver_input)
    max_level = option_number(options, "-max-levels",
                              option_number(options, "-start-levels", 8))
    resolution = _resolution_for_max_level(max_level)

    # Serial: a cells-only read is milliseconds per snapshot, so a pool
    # here was machinery without a payoff.
    files = snapshot_files(snapshot_dir)
    records = [read_adaptive_grid(f) for f in files]

    # Each record is read_adaptive_grid's 5-tuple
    # (cells, domain_min, domain_max, time, num_cells); [3] is time.
    order = sorted(range(len(records)), key=lambda i: records[i][3])
    records = [records[i] for i in order]
    return {
        "records": records,
        "times": np.array([r[3] for r in records], dtype=float),
        "dofs": np.array([r[4] for r in records], dtype=int),
        "resolution": resolution,
        "max_level": int(max_level),
    }


# One size for the still and every movie frame -- stated once.
FIGSIZE = (6.5, 4.2)


def draw_level_frame(fig, ax, record, frame_index, frame_count, data):
    """One refinement-level profile: a filled step curve versus velocity."""
    cells, dmin, dmax, time, num_cells = record
    v, level = refinement_levels(cells, dmin, dmax, data["resolution"])
    ax.fill_between(v, level, step="mid", alpha=0.35, color="#2d1e8f")
    ax.step(v, level, where="mid", color="#2d1e8f", lw=1.5)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(0, data["max_level"] + 1)
    ax.set_xlabel(r"$v/v_{th}$")
    ax.set_ylabel("local refinement level")
    ax.set_title(
        "Adaptive grid refinement\n"
        + rf"$t={time:.2f}\,\tau$,  {num_cells} cells "
        + f"({frame_index + 1}/{frame_count})",
        fontsize=12,
    )
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.14, top=0.82)


def plot_dof(data):
    """Active cell count versus simulation time."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    line1d(
        ax, data["times"],
        [(data["dofs"], None, {"color": "#2d1e8f", "lw": 2.0})],
        scale="linear", legend=False,
    )
    ax.set_xlabel(r"simulation time [$\tau$]")
    ax.set_ylabel("active cells")
    ax.set_title("Adaptive sparse-grid size")
    ax.grid(alpha=0.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description="ICRF_1D adaptive-grid plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("-n", "--points", type=int, default=None,
                        help=argparse.SUPPRESS)  # uniform flag surface
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    data = load(args.output)

    def draw(fig, ax, index):
        draw_level_frame(fig, ax, data["records"][index], index,
                         len(data["records"]), data)

    if do_static:
        save_png(render_still(draw, len(data["records"]) - 1, figsize=FIGSIZE),
                 args.fig_dir, "grid_level", dpi=220)
        save_png(plot_dof(data), args.fig_dir, "grid_dof", dpi=220)
    if do_movie:
        render_movie(draw, len(data["records"]),
                     str(Path(args.fig_dir) / "grid_level.mp4"),
                     figsize=FIGSIZE, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
