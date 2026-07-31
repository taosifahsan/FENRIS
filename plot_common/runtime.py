"""Process setup shared by every plotter: backend choice + project paths.

Every ``plot/*.py`` entry point calls :func:`bootstrap` before importing
matplotlib.

There is no parallel machinery here (or anywhere in the plotting stack):
everything runs in one serial process, and the only concurrency is asgard's
own OpenMP inside snapshot reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved data locations for one project.

    Frozen because these are facts about the on-disk layout, not settings: a
    plotter that wants to write somewhere else takes a ``--fig-dir`` argument
    rather than mutating this.
    """

    snapshots: Path      # output_data/movie -- the per-step HDF5 series
    tables: Path         # output_data/tables -- ICRF's .bin coefficient tables
    figures: Path
    solver_input: Path   # input_data/input_solver.txt
    build_input: Path    # input_data/input_build.txt (ICRF only)


def _resolve_paths(caller_file) -> ProjectPaths:
    """Derive the project layout from a plotter's ``__file__``.

    The project root is the parent of the directory the caller lives in:
    ``<project>/plot/solution.py`` -> ``<project>/``.  Deriving it this way
    (rather than searching upward for a marker file) keeps the rule trivial to
    state and means a misplaced file fails loudly instead of silently resolving
    to some ancestor directory.
    """
    root = Path(caller_file).resolve().parent.parent
    solver_input = root / "input_data" / "input_solver.txt"
    # The snapshot directory is the deck's ``movie_dir`` key, exactly as the
    # solver reads it (same default) -- the deck is the single source of
    # truth, so renaming the directory there cannot silently leave the
    # plotters reading a stale location.
    from plot_common.reader import read_options  # deferred: keeps import order trivial
    movie_dir = "movie"
    if solver_input.is_file():
        movie_dir = read_options(solver_input).get("movie_dir", "movie")
    return ProjectPaths(
        snapshots=root / "output_data" / movie_dir,
        tables=root / "output_data" / "tables",
        figures=root / "figures",
        solver_input=solver_input,
        build_input=root / "input_data" / "input_build.txt",
    )


def bootstrap(caller_file) -> ProjectPaths:
    """Select the file-only matplotlib backend and resolve project paths.

    Call this first, before importing matplotlib or anything that imports it:
    matplotlib reads ``MPLBACKEND`` once, at import time.  Agg is the backend
    that renders straight to files with no display, which is all these batch
    plotters ever do -- and it works identically on a laptop and on a
    display-less cluster node.
    """
    os.environ["MPLBACKEND"] = "Agg"
    return _resolve_paths(caller_file)
