"""The adaptive sparse grid: refinement-level map and degrees-of-freedom count.

Does **not** use :class:`~plot_common.reader.SnapshotCache`.  A grid frame
needs only ``snapshot.cells`` -- the active hierarchical cell indices -- never
a dense reconstruction, so reading it is dramatically cheaper than the
solution/smoothed/diagnostics pipeline.  That is why this file does its own
lightweight read rather than sharing the expensive cache: forcing it through
the shared cache would gain nothing and complicate the one thing that is
allowed to run concurrently with the expensive reconstruction pass in
``show_all.py``.

Two outputs:

1. **Refinement-level map** (static: final frame; movie: one frame per
   snapshot) -- a filled contour of "how many times has the grid been refined
   here", via :func:`plot_common.static.level_contour`.
2. **Degrees-of-freedom history** -- active cell count versus simulation time,
   a plain linear line plot.

The refinement-level geometry (:func:`hierarchical_interval`,
:func:`refinement_levels`) lives here rather than in
``plot_common/calculations.py`` because this is its only consumer -- keeping
it local means the whole algorithm is visible in one file rather than split
across a shared module and a caller.

Used by: ``ICRF_2D/plot/show_all.py``.

Depends on: :mod:`plot_common.reader` (``read_adaptive_grid``),
:mod:`plot_common.static` (``level_contour``, ``line1d``),
:mod:`plot_common.movie` (parallel frame rendering).
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
    option_vector,
    read_adaptive_grid,
    read_options,
    snapshot_files,
)
from plot_common.runtime import process_pool, worker_count
from plot_common.static import level_contour, line1d, save_png


# ---------------------------------------------------------------------------
# Refinement-level geometry
# ---------------------------------------------------------------------------
#
# ASGarD identifies each 1-D basis cell by a single integer index in a
# hierarchical (binary-tree) numbering:
#
#   index 0, 1  -> the whole domain (root level)
#   index 2, 3  -> the two halves
#   index 4..7  -> the four quarters, and so on
#
# So the *level* of an index is ``bit_length() - 1``, and the cell's extent on
# the normalized [0, 1] interval follows from its position within its level.


def hierarchical_interval(index, domain_left, domain_right):
    """Return the physical ``(left, right)`` span of one hierarchical cell.

    Indices 0 and 1 are the root level and cover the entire domain.

    For deeper indices: ``scale`` is ``2**level``, obtained via ``bit_length``
    (the position of the highest set bit gives the level directly and avoids
    a log). Within a level, index ``p`` covers
    ``[p/scale - 1, (p+1)/scale - 1]`` in normalized coordinates -- the ``-1``
    shift accounts for the level's indices starting at ``scale`` rather than
    0.  That normalized span is then mapped onto the physical domain.
    """
    if index <= 1:
        return domain_left, domain_right

    scale = 1 << (int(index).bit_length() - 1)
    unit_left = index / scale - 1.0
    unit_right = (index + 1) / scale - 1.0
    width = domain_right - domain_left
    return domain_left + width * unit_left, domain_left + width * unit_right

def _span_lookup(indices, samples, domain_left, domain_right):
    """Per distinct index: the grid slice its cell covers, and its level.

    Returns ``{index: (first, last, level)}``.  Computed once per *index*
    rather than once per *cell*, which is the whole point -- a frame has
    thousands of cells but only a couple of hundred distinct indices per
    dimension.

    ``side="left"`` / ``side="right"`` together reproduce the inclusive
    ``(coord >= left) & (coord <= right)`` selection for an ascending vector,
    so a point sitting exactly on a cell boundary counts as inside both
    neighbours -- which is what makes the drawn bands meet cleanly.

    A dimension's level is ``bit_length() - 1``; indices 0 and 1 both give 0,
    since the root-to-level-one transition covers the same extent and so adds
    no physical refinement.
    """
    spans = {}
    for index in np.unique(indices):
        index = int(index)
        left, right = hierarchical_interval(index, domain_left, domain_right)
        spans[index] = (
            int(np.searchsorted(samples, left, side="left")),
            int(np.searchsorted(samples, right, side="right")),
            max(0, index.bit_length() - 1),
        )
    return spans



SAMPLES_PER_LEAF = 4


def _resolution_for_max_level(max_level, cap=2049):
    """Sample count matched to the run's finest possible cell.

    A level-``L`` cell is ``1/2**L`` of the (normalized) domain wide, so
    ``2**L`` leaves tile the domain and ``SAMPLES_PER_LEAF * 2**L + 1``
    evenly spaced points give every deepest-possible cell a fixed handful of
    interior samples (the ``+1`` closes the right edge).  That guarantees no
    active cell is ever narrower than the sampling -- the old hardcoded count
    silently under-resolved decks whose ``-max-levels`` exceeded it -- while a
    coarse run no longer pays for resolution it cannot contain.

    ``SAMPLES_PER_LEAF`` exists purely for the drawing: ``contourf``
    interpolates between samples, so with only one sample per leaf a band edge
    smears across a whole cell width; a few samples per leaf pin it to a
    fraction of that.

    Derived from the deck's ``-max-levels`` (the run's ceiling), not from
    whichever level happens to be active in one frame, so the sampling grid is
    identical across every frame of a movie.  The cap only guards a
    hand-edited deck with an absurd level; everything this repo uses sits far
    below it.
    """
    return min(cap, SAMPLES_PER_LEAF * (1 << max(0, int(max_level))) + 1)


def refinement_levels(cells, domain_min, domain_max, x_points, theta_points):
    """Map the deepest active refinement level covering each grid point.

    This produces the field behind the adaptive-grid ``contourf`` plot: an
    ``x_points x theta_points`` integer array whose value is how deeply the
    grid is refined there.  Returns ``(theta, x, level)`` ready for plotting.
    ``x_points``/``theta_points`` should come from :func:`_resolution_for_max_level`
    applied to each dimension's own ``-max-levels`` entry -- the two levels
    need not match, so neither does the resolution.

    The algorithm is one sentence: every active cell paints its own rectangle
    with its own level, and where cells overlap the deeper one wins.  Cells
    genuinely do overlap -- a snapshot lists every ancestor the hierarchical
    basis needs, not just the leaves -- which is why this is a ``maximum``
    rather than an assignment.

    Two things make it fast.  Each cell's rectangle is a *contiguous run* of
    both sorted coordinate vectors, so it is a plain 2-D slice rather than a
    fancy-index (building two full-length boolean masks per cell and combining
    them with ``np.ix_`` touches the whole grid once per cell, for thousands of
    cells).  And each cell's span is looked up rather than recomputed: there
    are only a couple of hundred distinct indices across thousands of cells,
    so :func:`_span_lookup` does the interval arithmetic and ``searchsorted``
    once per index instead of once per cell.
    """
    x = np.linspace(domain_min[0], domain_max[0], x_points)
    theta = np.linspace(domain_min[1], domain_max[1], theta_points)
    level = np.zeros((x_points, theta_points), dtype=int)

    # Deduplicate: the same ancestor can be listed via several hierarchy
    # paths, and repainting it cannot change the result.
    unique_cells = np.unique(np.asarray(cells, dtype=int), axis=0)
    x_span = _span_lookup(unique_cells[:, 0], x, domain_min[0], domain_max[0])
    theta_span = _span_lookup(unique_cells[:, 1], theta, domain_min[1], domain_max[1])

    for x_index, theta_index in unique_cells:
        x_first, x_last, x_level = x_span[x_index]
        theta_first, theta_last, theta_level = theta_span[theta_index]
        # With deck-derived resolution every active cell spans several
        # samples; an empty run is possible only when the resolution cap
        # truncated an absurdly deep deck.
        if x_first >= x_last or theta_first >= theta_last:
            continue
        block = level[x_first:x_last, theta_first:theta_last]
        np.maximum(block, x_level + theta_level, out=block)
    return theta, x, level



# ---------------------------------------------------------------------------
# Reading and deriving
# ---------------------------------------------------------------------------


def load(snapshot_dir, workers=None, solver_input=None):
    """Read the grid structure of every snapshot, in parallel.

    Cheap: each task reads only ``snapshot.cells`` and two domain vectors, no
    reconstruction.  Returns a dict with ``records`` (one
    ``(cells, domain_min, domain_max, time, num_cells)`` tuple per snapshot,
    time-sorted), ``times``, ``dofs`` (active cell count history), and
    ``resolution`` -- the per-dimension sampling counts derived from the
    deck's ``-max-levels``, so the level map is exactly as fine as the run
    could ever have refined (see :func:`_resolution_for_max_level`).
    """
    max_levels = option_vector(
        read_options(solver_input or PATHS.solver_input), "-max-levels")
    resolution = (_resolution_for_max_level(max_levels[0]),
                  _resolution_for_max_level(max_levels[1]))

    files = snapshot_files(snapshot_dir)
    pool_width = min(worker_count(workers), max(1, len(files)))
    if pool_width == 1 or len(files) == 1:
        records = [read_adaptive_grid(f) for f in files]
    else:
        with process_pool(pool_width) as pool:
            records = list(pool.map(read_adaptive_grid, files, chunksize=8))

    # Each record is read_adaptive_grid's 5-tuple
    # (cells, domain_min, domain_max, time, num_cells); [3] is time, [4] is
    # num_cells.  Snapshot filenames don't guarantee time order, so sort here
    # once rather than let every caller re-derive the same ordering.
    order = sorted(range(len(records)), key=lambda i: records[i][3])
    records = [records[i] for i in order]
    times = np.array([r[3] for r in records], dtype=float)
    dofs = np.array([r[4] for r in records], dtype=int)
    return {"records": records, "times": times, "dofs": dofs,
            "resolution": resolution}


# ---------------------------------------------------------------------------
# Static figures
# ---------------------------------------------------------------------------


def draw_level_frame(fig, ax, record, frame_index, frame_count, resolution):
    """Draw one refinement-level map from an already-read grid record.

    ``resolution`` is the deck-derived ``(x_points, theta_points)`` pair from
    :func:`load`, fixed for the whole run so every movie frame samples the
    same grid.
    """
    cells, domain_min, domain_max, time, _num_cells = record
    theta_min, theta_max = np.degrees(domain_min[1]), np.degrees(domain_max[1])
    x_min, x_max = domain_min[0], domain_max[0]
    theta, x, level = refinement_levels(cells, domain_min, domain_max,
                                        resolution[0], resolution[1])

    level_contour(ax, np.degrees(theta), x, level, fig=fig)
    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel(r"$\theta_0$ [deg]")
    ax.set_ylabel(r"$x_0$")
    ax.set_title(
        "Adaptive sparse-grid refinement\n"
        + rf"$t={time:.3f}\,\tau_c$ ({frame_index + 1}/{frame_count})",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.14, top=0.84)


def plot_static(data):
    """Final-frame refinement-level map."""
    records = data["records"]
    fig = plt.figure(figsize=(5.4, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    draw_level_frame(fig, ax, records[-1], len(records) - 1, len(records),
                     data["resolution"])
    return fig


def plot_dof(data):
    """Active cell count (degrees of freedom) versus simulation time."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    line1d(
        ax, data["times"],
        [(data["dofs"], None, {"color": "#2d1e8f", "lw": 2.0})],
        scale="linear", legend=False,
    )
    ax.set_xlabel(r"simulation time [$\tau_c$]")
    ax.set_ylabel("active cells")
    ax.set_title("Adaptive sparse-grid size")
    ax.grid(alpha=0.25)
    return fig


# ---------------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------------

# Per-worker state: the records list and sampling resolution, sent once via
# the pool initializer rather than once per frame.
_RECORDS = None
_RESOLUTION = None


def _init_grid_worker(records, resolution):
    global _RECORDS, _RESOLUTION
    _RECORDS, _RESOLUTION = records, resolution


def _draw_grid_frame_task(task):
    """Worker: draw and save one refinement-level frame.

    Module-level (not a closure) so it can be pickled for ``spawn`` workers;
    receives the shared records list via the pool initializer.

    Deliberately **no** ``bbox_inches="tight"``: a fixed ``figsize x dpi``
    keeps every frame's pixel dimensions identical (and even), which H.264's
    ``yuv420p`` requires.  ``bbox_inches="tight"`` crops to content and would
    make dimensions vary -- and occasionally come out odd -- per frame,
    silently breaking the H.264 encode and falling back to unplayable MPEG-4.
    """
    index = task["index"]
    fig = plt.figure(figsize=(5.4, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    draw_level_frame(fig, ax, _RECORDS[index], index, len(_RECORDS),
                     _RESOLUTION)
    fig.savefig(f"{task['frame_dir']}/frame_{index:06d}.png", dpi=task["dpi"])
    plt.close(fig)


def plot_movie(data, output_file, *, workers=None, fps=8, dpi=140):
    """Render the refinement-level movie, one frame per snapshot."""
    records = data["records"]
    return render_movie(
        _draw_grid_frame_task, len(records), output_file,
        fps=fps, dpi=dpi, workers=workers,
        initializer=_init_grid_worker,
        initargs=(records, data["resolution"]),
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ICRF adaptive-grid plots")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("--fig-dir", default=str(PATHS.figures))
    parser.add_argument("-j", "--workers", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    # Accepted and ignored: this plotter reads only the grid structure, never
    # a reconstruction, so resolution does not apply.  It is accepted anyway
    # so that a script can pass the same flags to every plotter uniformly.
    parser.add_argument("-n", "--points", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)

    data = load(args.output, workers=args.workers)

    if do_static:
        save_png(plot_static(data), args.fig_dir, "grid_level", dpi=220)
        save_png(plot_dof(data), args.fig_dir, "grid_dof", dpi=220)
    if do_movie:
        out = str(Path(args.fig_dir) / "grid_level.mp4")
        plot_movie(data, out, workers=args.workers, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
