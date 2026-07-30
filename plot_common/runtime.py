"""Process setup and worker-pool primitives shared by every plotter.

One coherent job: get this process ready to plot (thread pinning, matplotlib
backend, project data paths) and hand out worker pools.  Every ``plot/*.py``
entry point calls :func:`bootstrap` before importing matplotlib.

No venv re-exec: the orchestration shell scripts (``tools/run_*.sh``) activate
the correct venv before invoking Python, so there is nothing for this module to
detect or correct.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import tempfile


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

    name: str            # "ICRF_2D" or "LHCD_2D"
    root: Path
    input_data: Path
    output_data: Path
    snapshots: Path      # output_data/movie -- the per-step HDF5 series
    tables: Path         # output_data/tables -- ICRF's .bin coefficient tables
    figures: Path
    solver_input: Path   # input_data/input_solver.txt
    build_input: Path    # input_data/input_build.txt (ICRF only)
    final_output: Path   # output_data/output.h5

    @property
    def has_tables(self) -> bool:
        """True when this project ships precomputed coefficient tables.

        Distinguishes ICRF (table-driven coefficients) from LHCD (analytic),
        without either project's plotters having to name the other.
        """
        return self.tables.is_dir()

    @property
    def has_build_input(self) -> bool:
        """True when a build deck exists (ICRF has one; LHCD does not)."""
        return self.build_input.is_file()


def _resolve_paths(caller_file) -> ProjectPaths:
    """Derive the project layout from a plotter's ``__file__``.

    The project root is the parent of the directory the caller lives in:
    ``<project>/plot/solution.py`` -> ``<project>/``.  Deriving it this way
    (rather than searching upward for a marker file) keeps the rule trivial to
    state and means a misplaced file fails loudly instead of silently resolving
    to some ancestor directory.
    """
    plot_dir = Path(caller_file).resolve().parent
    root = plot_dir.parent
    input_data = root / "input_data"
    output_data = root / "output_data"
    return ProjectPaths(
        name=root.name,
        root=root,
        input_data=input_data,
        output_data=output_data,
        snapshots=output_data / "movie",
        tables=output_data / "tables",
        figures=root / "figures",
        solver_input=input_data / "input_solver.txt",
        build_input=input_data / "input_build.txt",
        final_output=output_data / "output.h5",
    )


# ---------------------------------------------------------------------------
# Bootstrap: must run before matplotlib is imported
# ---------------------------------------------------------------------------
#
# Thread pinning and the plotting backend are both environment-variable
# settings matplotlib (and numpy's BLAS backend) reads once, at import time --
# so bootstrap() must be the first thing a plotter does, before any other
# import that might pull in matplotlib transitively.


def _pin_native_threads() -> None:
    """One native math thread per process.

    Parallelism comes entirely from the process pool in this module.  If each
    worker also spawned a full-width OpenMP/BLAS thread team, a 32-worker pool
    on a 32-core node would run 1024 threads and thrash rather than compute.
    """
    threads = os.environ.get("PLOT_NATIVE_THREADS", "1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = threads


def _select_backend(force_agg: bool) -> None:
    """Choose the matplotlib backend before matplotlib is imported.

    These plotters never call ``plt.show()`` -- every figure is written to
    disk, whether by a worker process or the parent -- so Agg is essentially
    always the right choice.  ``force_agg`` states that outright; otherwise the
    absence of a display is treated as the same signal.

    Forcing this matters beyond cosmetics: on macOS, instantiating a GUI
    backend window off the main thread **aborts the interpreter** rather than
    raising a catchable exception. A worker process or a background thread that
    accidentally pulls in a GUI backend takes the whole run down.
    """
    if force_agg or not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ["MPLBACKEND"] = "Agg"
    else:
        os.environ.setdefault("MPLBACKEND", "Agg")


def bootstrap(caller_file, *, force_agg: bool = True) -> ProjectPaths:
    """Pin native threads, choose the mpl backend, and resolve project paths.

    Call this first, before importing matplotlib or anything that imports it.
    Returns the :class:`ProjectPaths` for the caller's project.

    ``force_agg`` defaults to True: these plotters are batch tools, not
    interactive viewers, and always want the file backend.
    """
    _pin_native_threads()
    paths = _resolve_paths(caller_file)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), f"{paths.name.lower()}_matplotlib"),
    )
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    _select_backend(force_agg)
    return paths


# ---------------------------------------------------------------------------
# Worker pools
# ---------------------------------------------------------------------------


def worker_count(requested=None):
    """Resolve a worker count: explicit request, else the CPU allocation.

    Checks Slurm's own view of the allocation before falling back to
    ``os.cpu_count()``, so a job that requested fewer cores than the node has
    does not oversubscribe itself.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return max(1, os.cpu_count() or 1)


def process_pool(workers, **kwargs):
    """Create a process pool using fresh spawned interpreters.

    ``spawn`` rather than the platform default ``fork``: forking a process that
    already has matplotlib and HDF5 loaded is unsafe (both hold internal state,
    locks, and open handles that do not survive a fork cleanly), and ``spawn``
    is the only start method that behaves consistently across Linux and macOS.
    """
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=multiprocessing.get_context("spawn"),
        **kwargs,
    )
