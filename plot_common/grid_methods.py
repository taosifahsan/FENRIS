"""The adaptive sparse grid: refinement level and DOF count, 1-D and 2-D.

One implementation shared by all four projects.  The four ``plot/grid.py``
files were 96-99% identical -- they differed only in axis labels and the
argparse description -- so they are now thin adapters that call
:func:`main_1d` or :func:`main_2d` with those strings.

Reads only ``snapshot.cells`` (the active hierarchical indices), never a
reconstruction, so this is far cheaper than the solution pipeline.

Two figures per project:

1. **Refinement-level profile** (static: final frame; movie: one frame per
   snapshot).  In 2-D a filled contour over ``(theta, x)``; in 1-D that
   collapses to a filled step curve versus velocity.
2. **Degrees-of-freedom history** -- active cell count versus time.

Used by: each project's ``plot/grid.py``, in turn by ``tools/run.sh``.

Depends on: :mod:`plot_common.reader` (``read_adaptive_grid``, deck),
:mod:`plot_common.static` (``level_contour``, ``line1d``, ``save_png``),
:mod:`plot_common.movie` (movie rendering).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common.movie import render_movie
from plot_common.reader import (
    option_number,
    option_vector,
    read_adaptive_grid,
    read_options,
    snapshot_files,
)
from plot_common.static import level_contour, line1d, render_still, save_png


# ---------------------------------------------------------------------------
# Refinement-level geometry (shared by both dimensionalities)
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
    active cell is ever narrower than the sampling -- a hardcoded count
    silently under-resolves decks whose ``-max-levels`` exceeds it -- while a
    coarse run no longer pays for resolution it cannot contain.

    ``SAMPLES_PER_LEAF`` exists purely for the drawing: ``contourf``
    interpolates between samples, so with only one sample per leaf a band edge
    smears across a whole cell width; a few samples per leaf pin it to a
    fraction of that.

    Derived from the deck's ``-max-levels`` (the run's ceiling), not from
    whichever level happens to be active in one frame, so the sampling grid is
    identical across every frame of a movie.  The cap only guards a
    hand-edited deck with an absurd level; ``cap`` is generous in 1-D, where
    sampling is a vector rather than a plane and so costs almost nothing.
    """
    return min(cap, SAMPLES_PER_LEAF * (1 << max(0, int(max_level))) + 1)


# ---------------------------------------------------------------------------
# One dimension
# ---------------------------------------------------------------------------


def refinement_levels_1d(cells, domain_min, domain_max, points):
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


def load_1d(snapshot_dir, solver_input):
    """Read the grid structure of every snapshot (cheap, serial).

    Returns records (time-sorted ``(cells, dmin, dmax, time, num_cells)``),
    ``times``, ``dofs``, and the deck-derived sampling ``resolution``.
    """
    options = read_options(solver_input)
    max_level = option_number(options, "-max-levels",
                              option_number(options, "-start-levels", 8))
    resolution = _resolution_for_max_level(max_level, cap=32769)

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


FIGSIZE_1D = (6.5, 4.2)
FIGSIZE_2D = (5.4, 4.5)


def draw_level_frame_1d(fig, ax, record, frame_index, frame_count, data,
                        xlabel, time_label):
    """One refinement-level profile: a filled step curve versus velocity."""
    cells, dmin, dmax, time, num_cells = record
    v, level = refinement_levels_1d(cells, dmin, dmax, data["resolution"])
    ax.fill_between(v, level, step="mid", alpha=0.35, color="#2d1e8f")
    ax.step(v, level, where="mid", color="#2d1e8f", lw=1.5)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(0, data["max_level"] + 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("local refinement level")
    ax.set_title(
        "Adaptive grid refinement\n"
        + rf"$t={time:.2f}\,{time_label}$,  {num_cells} cells "
        + f"({frame_index + 1}/{frame_count})",
        fontsize=12,
    )
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.14, top=0.82)


# ---------------------------------------------------------------------------
# Two dimensions
# ---------------------------------------------------------------------------


def refinement_levels_2d(cells, domain_min, domain_max, x_points, theta_points):
    """Map the deepest active refinement level covering each grid point.

    This produces the field behind the adaptive-grid ``contourf`` plot: an
    ``x_points x theta_points`` integer array whose value is how deeply the
    grid is refined there.  Returns ``(theta, x, level)`` ready for plotting.
    ``x_points``/``theta_points`` should come from
    :func:`_resolution_for_max_level` applied to each dimension's own
    ``-max-levels`` entry -- the two levels need not match, so neither does
    the resolution.

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


def load_2d(snapshot_dir, solver_input):
    """Read the grid structure of every snapshot (cheap, serial).

    Cheap: each task reads only ``snapshot.cells`` and two domain vectors, no
    reconstruction.  Returns a dict with ``records`` (one
    ``(cells, domain_min, domain_max, time, num_cells)`` tuple per snapshot,
    time-sorted), ``times``, ``dofs`` (active cell count history), and
    ``resolution`` -- the per-dimension sampling counts derived from the
    deck's ``-max-levels``, so the level map is exactly as fine as the run
    could ever have refined (see :func:`_resolution_for_max_level`).
    """
    max_levels = option_vector(read_options(solver_input), "-max-levels")
    resolution = (_resolution_for_max_level(max_levels[0]),
                  _resolution_for_max_level(max_levels[1]))

    # Serial: a cells-only read is milliseconds per snapshot, so a pool
    # here was machinery without a payoff.
    files = snapshot_files(snapshot_dir)
    records = [read_adaptive_grid(f) for f in files]

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


def draw_level_frame_2d(fig, ax, record, frame_index, frame_count, resolution,
                        xlabel, ylabel, time_label):
    """Draw one refinement-level map from an already-read grid record.

    ``resolution`` is the deck-derived ``(x_points, theta_points)`` pair from
    :func:`load_2d`, fixed for the whole run so every movie frame samples the
    same grid.
    """
    cells, domain_min, domain_max, time, _num_cells = record
    theta_min, theta_max = np.degrees(domain_min[1]), np.degrees(domain_max[1])
    x_min, x_max = domain_min[0], domain_max[0]
    theta, x, level = refinement_levels_2d(cells, domain_min, domain_max,
                                           resolution[0], resolution[1])

    level_contour(ax, np.degrees(theta), x, level, fig=fig)
    ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Adaptive sparse-grid refinement\n"
        + rf"$t={time:.3f}\,{time_label}$ ({frame_index + 1}/{frame_count})",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.14, top=0.84)


# ---------------------------------------------------------------------------
# Shared figure and CLI
# ---------------------------------------------------------------------------


def plot_dof(data, time_label):
    """Active cell count (degrees of freedom) versus simulation time."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    line1d(
        ax, data["times"],
        [(data["dofs"], None, {"color": "#2d1e8f", "lw": 2.0})],
        scale="linear", legend=False,
    )
    ax.set_xlabel(rf"simulation time [${time_label}$]")
    ax.set_ylabel("active cells")
    ax.set_title("Adaptive sparse-grid size")
    ax.grid(alpha=0.25)
    return fig


def _parse(description, paths):
    """The flag surface every plotter shares.

    ``-n/--points`` is accepted and ignored: this plotter reads only the grid
    structure, never a reconstruction, so resolution does not apply.  It is
    accepted anyway so a script can pass the same flags to every plotter.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("-o", "--output", default=str(paths.snapshots))
    parser.add_argument("--fig-dir", default=str(paths.figures))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("-n", "--points", type=int, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def _render(args, data, draw, figsize, time_label):
    """Shared static/movie dispatch.

    Giving neither ``--static`` nor ``--movie`` renders both -- that is how
    tools/run.sh invokes every plotter; either flag narrows a manual run to
    just that output.
    """
    do_static = args.static or not (args.static or args.movie)
    do_movie = args.movie or not (args.static or args.movie)
    if do_static:
        save_png(render_still(draw, len(data["records"]) - 1, figsize=figsize),
                 args.fig_dir, "grid_level", dpi=220)
        save_png(plot_dof(data, time_label), args.fig_dir, "grid_dof", dpi=220)
    if do_movie:
        render_movie(draw, len(data["records"]),
                     str(Path(args.fig_dir) / "grid_level.mp4"),
                     figsize=figsize, fps=args.fps, dpi=args.dpi)


def main_1d(paths, description, xlabel, time_label=r"\tau"):
    """CLI entry point for a 1-D project's ``plot/grid.py``."""
    args = _parse(description, paths)
    data = load_1d(args.output, paths.solver_input)

    # Bind the derived data into the (fig, ax, index) signature render_still
    # and render_movie expect (closures are fine: rendering is in-process).
    def draw(fig, ax, index):
        draw_level_frame_1d(fig, ax, data["records"][index], index,
                            len(data["records"]), data, xlabel, time_label)

    _render(args, data, draw, FIGSIZE_1D, time_label)


def main_2d(paths, description, xlabel, ylabel, time_label=r"\tau_c"):
    """CLI entry point for a 2-D project's ``plot/grid.py``."""
    args = _parse(description, paths)
    data = load_2d(args.output, paths.solver_input)

    def draw(fig, ax, index):
        draw_level_frame_2d(fig, ax, data["records"][index], index,
                            len(data["records"]), data["resolution"],
                            xlabel, ylabel, time_label)

    _render(args, data, draw, FIGSIZE_2D, time_label)
