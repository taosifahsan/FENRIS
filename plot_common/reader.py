"""**Everything that reads from disk**, and the cache that holds the result.

The only module in the plotting stack that touches the filesystem for *input*.
If you are looking for "how does a number get from a file into an array", it is
here.  Nothing in this file computes physics or draws anything.

Shared by both projects because snapshot reading is a property of the *solver*
(ASGarD's sparse-grid file format), not of the physics being solved -- ICRF and
LHCD snapshots are read identically.

Three kinds of input, one section each below:

1. **HDF5 solver snapshots** (``*.h5``), one per output step, holding the
   sparse-grid representation of the distribution function.  The ``asgard``
   module reconstructs them into dense arrays.
2. **Input decks** (``input_solver.txt``, ``input_build.txt``) -- flat
   ``key : value`` text.  Both the solver and the plotters read them, so the
   plotting side must parse them the same way.
3. **Raw binary tables** (``*.bin``) -- headerless float64 dumps from ICRF's
   ``build_tables.cpp``.  Their shape comes from ``parameters.bin``; this module
   reads the bytes, and ``ICRF_2D/plot/coefficients.py`` interprets them.

Section 4 is the important one architecturally: :class:`SnapshotCache` and
:func:`load_snapshots`.

**Why a cache exists at all.**  Reconstruction (``read_snapshot``) is by far the
most expensive operation in the pipeline -- it evaluates every active basis
function at every grid point, at a cost scaling as ``points**2``.  Five separate
plots need the *same* reconstruction of the *same* snapshot: the solution field,
both smoothed marginals, the growth rate, and the conserved moments.  Reading
once into memory and sharing costs about 29 MB at default settings
(``points=192``, ~200 snapshots) and removes a five-fold duplication of the only
genuinely slow step.

That single fact is what lets every plotter stay simple: they receive arrays,
not filenames.

Used by: every plotter in ``ICRF_2D/plot/`` and ``LHCD_2D/plot/``.

Depends on: ``asgard`` (solver Python bindings), ``numpy``,
``plot_common.parallel`` (the load is parallel).  Deliberately does *not* import
matplotlib -- reading data must never require a display stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re

import asgard
import numpy as np

from plot_common.runtime import process_pool, worker_count


# ---------------------------------------------------------------------------
# Section 1: HDF5 solver snapshots
# ---------------------------------------------------------------------------
#
# A run writes many snapshot files into one directory.  Their names embed the
# step number (``..._step_37.h5``), and every stage of the plotting pipeline
# needs them in *simulation time* order, which for these runs is the same as
# step order.


def natural_key(path):
    """Sort key that orders ``snapshot_9`` before ``snapshot_10``.

    Plain lexicographic sorting compares strings character by character, so
    ``"10"`` sorts before ``"9"`` (because ``"1" < "9"``).  That would shuffle
    movie frames into the wrong order.

    The fix is to split the filename on runs of digits and convert those runs
    to integers, so the comparison happens numerically where it matters:

        "snap_10.h5" -> ["snap_", 10, ".h5"]
        "snap_9.h5"  -> ["snap_", 9,  ".h5"]

    Python then compares element by element and gets ``9 < 10`` as intended.
    ``os.path.basename`` keeps directory names out of the comparison, so files
    from different directories still sort by their own names.
    """
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(part) if part.isdigit() else part for part in parts]


def snapshot_files(path):
    """Return the time-ordered snapshot list for ``path``.

    ``path`` may be either:
      - a single snapshot file, in which case a one-element list comes back
        (this is what makes ``--output some_snapshot.h5`` work for single
        figures), or
      - a directory of snapshots, which is globbed and sorted.

    Both ``.h5`` and ``.hdf5`` extensions are accepted because ASGarD's
    extension depends on how it was built.  ``set()`` removes the duplicates
    that a directory containing both spellings of the same file would produce.

    Raises ``FileNotFoundError`` on an empty directory rather than returning an
    empty list, because every caller treats "no snapshots" as fatal -- failing
    here produces a clearer message than an ``IndexError`` deep in a movie
    loop later.
    """
    if os.path.isfile(path):
        return [path]

    files = []
    for pattern in ("*.h5", "*.hdf5"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    files = sorted(set(files), key=natural_key)
    if not files:
        raise FileNotFoundError(f"no HDF5 snapshot files found in {path}")
    return files


def read_snapshot(filename, points):
    """Reconstruct one snapshot onto a dense ``points x points`` grid.

    The solver stores the distribution function in a *sparse* hierarchical
    basis, not as a dense array -- that is the whole point of the adaptive
    scheme.  Plotting needs a dense array, so ASGarD evaluates the basis on a
    uniform grid for us.

    ``plot_data2d(((), ()), num_points=points)`` asks for a full 2-D
    reconstruction: the ``((), ())`` argument means "do not fix either
    dimension to a slice value, give me both axes".  Returns
    ``(values, x_axis, y_axis)``.

    Cost note: this call is the single most expensive operation in the whole
    plotting pipeline (it evaluates every active basis function at every grid
    point), and its cost grows as ``points**2``.  That is why the pipeline
    works so hard to reconstruct each snapshot exactly once and share the
    result across every figure and movie variant that needs it.
    """
    snapshot = asgard.pde_snapshot(filename)
    return snapshot.plot_data2d(((), ()), num_points=points)


def read_snapshot_with_domain(filename, points):
    """Reconstruct a snapshot *and* return its time and domain bounds.

    Same reconstruction as :func:`read_snapshot`, but from a single file open it
    also hands back the metadata that the boundary-flux diagnostic needs:

        ``(values, x_axis, y_axis, time, domain_min, domain_max)``

    This exists purely to avoid re-opening the file: the diagnostic needs the
    solver's declared domain edge (to evaluate flux exactly at the wall, which
    the sample grid need not reach) and the snapshot time, and opening the HDF5
    three times per snapshot would triple the cost of the most expensive scan
    in the pipeline.
    """
    snapshot = asgard.pde_snapshot(filename)
    values, x_axis, y_axis = snapshot.plot_data2d(((), ()), num_points=points)
    return (
        values, x_axis, y_axis,
        float(snapshot.time),
        np.asarray(snapshot.dimension_min, dtype=float),
        np.asarray(snapshot.dimension_max, dtype=float),
    )


def read_adaptive_grid(filename):
    """Return the *active sparse-grid structure* of one snapshot.

    This is deliberately separate from :func:`read_snapshot`: it reads only the
    grid bookkeeping, never reconstructing the solution.  That makes it
    dramatically cheaper, which is why the adaptive-grid movie can run in its
    own worker lane alongside the expensive solution stages.

    Returns a 5-tuple:
      ``cells``       -- ``(num_cells, num_dimensions)`` integer array of
                         hierarchical indices for every currently active cell.
                         ``snapshot.cells`` arrives flat, so it is reshaped
                         using the snapshot's own dimension count.
      ``domain_min``  -- physical lower bound per dimension
      ``domain_max``  -- physical upper bound per dimension
      ``time``        -- simulation time of this snapshot
      ``num_cells``   -- active cell count (also derivable from ``cells``, but
                         reported by the solver directly and used for the
                         cell-count-versus-time history plot)
    """
    snapshot = asgard.pde_snapshot(filename)
    return (
        np.asarray(snapshot.cells, dtype=int).reshape(
            -1, int(snapshot.num_dimensions)
        ),
        np.asarray(snapshot.dimension_min, dtype=float),
        np.asarray(snapshot.dimension_max, dtype=float),
        float(snapshot.time),
        int(snapshot.num_cells),
    )


def read_adaptive_grid_summary(filename):
    """Return only ``(time, active_cell_count)`` for one snapshot.

    The cell-count-versus-time history plot needs two scalars per snapshot and
    nothing else.  Reading just those avoids materializing the full ``cells``
    index array for every file, which matters when scanning hundreds of
    snapshots.
    """
    snapshot = asgard.pde_snapshot(filename)
    return float(snapshot.time), int(snapshot.num_cells)


def snapshot_time(filename):
    """Return the simulation time recorded inside a snapshot, or ``None``.

    Preferred over inferring time from the filename, because it is what the
    solver actually stamped.  Returns ``None`` (rather than raising) when the
    field is absent, so callers can fall back to ``step * dt`` -- older
    snapshots predate the time field.
    """
    try:
        snapshot = asgard.pde_snapshot(filename)
        return float(snapshot.time)
    except Exception:
        # Any failure here (missing field, unreadable file, older format) means
        # "time unknown"; the caller's step*dt fallback handles it.
        return None


def snapshot_step(filename):
    """Extract the integer step number from a ``..._step_<n>.h5`` filename.

    Returns ``None`` when the pattern is absent, letting the caller fall back
    to the frame's position in the file list.
    """
    match = re.search(r"_step_(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None


def diagnostic_snapshot_files(path):
    """Return snapshots for the time-history diagnostics, in plain sort order.

    Deliberately distinct from :func:`snapshot_files`: the diagnostics
    de-duplicate and re-sort their records *by the time value read out of each
    file* afterwards, so the order this returns only needs to be
    deterministic, not physically meaningful.  Kept as its own function so
    that changing movie frame ordering can never silently perturb diagnostic
    history assembly.
    """
    if os.path.isfile(path):
        return [path]
    files = []
    for pattern in ("*.h5", "*.hdf5"):
        files.extend(glob.glob(os.path.join(path, pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no HDF5 snapshots found in {path}")
    return files


# ---------------------------------------------------------------------------
# Section 2: input decks (``key : value`` text)
# ---------------------------------------------------------------------------
#
# Both projects configure runs with a flat text file:
#
#     -degree      : 2
#     -max-levels  : 5 5
#     x_max        : 8.0        # trailing comments are allowed
#
# The plotting code reads the *same* file the solver read, so that figures
# always describe the run that actually happened rather than a re-typed copy
# of its parameters.


def read_options(path) -> dict[str, str]:
    """Parse an input deck into a ``{key: raw_value_string}`` dictionary.

    Values are left as *strings* on purpose.  Some keys hold a single number,
    some hold a whitespace-separated vector (``-max-levels : 5 5``), and some
    hold text -- so interpretation is the caller's job, via the
    ``option_*`` helpers below.

    Parsing rules, applied per line:
      - everything from a ``#`` onward is a comment and is discarded;
      - blank lines, and lines with no ``:``, are skipped (this tolerates
        section banners and stray notes in hand-edited decks);
      - only the *first* ``:`` splits key from value, so values may contain
        colons;
      - keys and values are stripped of surrounding whitespace.

    A later duplicate key overwrites an earlier one, matching the "last
    assignment wins" behaviour of the solver's own parser.
    """
    path = Path(path)
    options: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"missing input file: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        options[key.strip()] = value.strip()
    return options


def option_number(options: dict[str, str], key: str, default=None):
    """Return the *first* token of ``key`` as a float, or ``default``.

    Used for keys whose value is a vector but where only the leading component
    is wanted -- e.g. ``-max-levels : 5 5`` yields ``5.0``.  Also returns
    ``default`` for a present-but-empty value, which a hand-edited deck can
    easily produce.
    """
    value = options.get(key)
    if value is None or not value:
        return default
    return float(value.split()[0])


def option_float(options: dict[str, str], key: str, default=None) -> float:
    """Return ``key`` as a float, raising if it is missing with no default.

    The strictness is intentional: a physics parameter that silently defaults
    to zero would produce a plausible-looking but wrong figure.  Passing an
    explicit ``default`` opts into leniency for genuinely optional knobs.
    """
    if key in options:
        return float(options[key])
    if default is None:
        raise KeyError(f"missing required option {key!r}")
    return float(default)


def option_vector(options: dict[str, str], key: str) -> np.ndarray:
    """Return every whitespace-separated token of ``key`` as a float array.

    For genuinely vector-valued keys (per-species masses, charges, and so on).
    Raises ``KeyError`` on a missing key for the same reason as
    :func:`option_float`.
    """
    return np.array([float(token) for token in options[key].split()], dtype=float)


# ---------------------------------------------------------------------------
# Section 3: raw binary tables
# ---------------------------------------------------------------------------


def read_binary_array(directory, name):
    """Read a flat ``float64`` binary file into a 1-D array.

    ICRF's ``build_tables.cpp`` dumps precomputed coefficient tables as bare
    little-endian doubles with **no header** -- no shape, no dtype, no
    element count.  This reads the raw sequence; giving it meaning (element
    count, 1-D versus 2-D, which axis is which) is the job of
    ``ICRF_2D/src/coefficients.py``, which knows the table layout.

    Kept generic (rather than named ``read_table``) because it is simply "read
    a packed float64 array from disk", the item-2 primitive that every binary
    reader in either project builds on.
    """
    return np.fromfile(os.path.join(directory, name), dtype=np.float64)


# ---------------------------------------------------------------------------
# Section 4: the snapshot cache
# ---------------------------------------------------------------------------
#
# This section is the reason the plotters can stay simple.  Every plotter that
# needs reconstructed data takes a SnapshotCache and never opens a file itself.


@dataclass
class SnapshotCache:
    """Every reconstructed snapshot of one run, held in memory.

    Attributes
    ----------
    frames
        List of ``(points, points)`` float32 arrays -- the reconstructed
        distribution function, one per snapshot, in ascending time order.
        float32 rather than float64 because these are display data: the
        reconstruction's own accuracy is far coarser than float32's ~7
        significant digits, and it halves the memory.
    times
        Simulation time of each frame, ascending.
    x, y
        The two coordinate axes shared by every frame.  Stored once rather than
        per frame because the reconstruction grid is fixed for a whole run --
        only the values change between snapshots.
    files
        Source path of each frame, kept for figure provenance and error
        messages.
    points
        Reconstruction resolution actually used.

    Notes
    -----
    Memory is ``points**2 * 4 * len(frames)`` bytes: about 29 MB at the default
    ``points=192`` with 200 snapshots, ~200 MB at ``points=512``.
    :func:`load_snapshots` prints this and warns past a threshold, so a run that
    would exhaust the node's memory says so rather than being OOM-killed.
    """

    frames: list
    times: list
    x: object
    y: object
    files: list
    points: int

    def __len__(self):
        return len(self.frames)

    @property
    def nbytes(self):
        """Total bytes held by the frame arrays."""
        return sum(frame.nbytes for frame in self.frames)

    def frame_pairs(self):
        """Yield ``(index, previous, current, dt)`` for consecutive frames.

        The growth rate is the only derived quantity needing two snapshots at
        once.  Yielding pairs from the cache is what makes that a free
        operation: the naive alternative -- each task reading both of its
        snapshots -- reconstructs every interior snapshot twice.

        Pairs whose time does not advance are skipped, since a restarted run can
        duplicate an output step and a zero ``dt`` would divide by zero.
        """
        for index in range(len(self.frames) - 1):
            dt = self.times[index + 1] - self.times[index]
            if dt <= 0.0:
                continue
            yield index, self.frames[index], self.frames[index + 1], dt


def _snapshot_load_task(task):
    """Worker: reconstruct one snapshot and return it with its time and axes.

    Returns the index too, because pool completion order is not task order and
    the cache must end up in time order.

    Casting to float32 here rather than in the parent means the array is
    already halved before it is pickled back, which is the dominant cost of
    returning it.
    """
    values, x_axis, y_axis = read_snapshot(task["path"], task["points"])
    time = snapshot_time(task["path"])
    if time is None:
        # Older snapshots predate the time field; fall back to step * dt, with
        # the frame index standing in for an unparseable filename.
        step = snapshot_step(task["path"])
        if step is None:
            step = task["index"]
        time = step * task["dt"]
    return task["index"], np.asarray(values, dtype=np.float32), x_axis, y_axis, float(time)


def load_snapshots(path, points, workers=None, dt=1.0, warn_bytes=2_000_000_000):
    """Read every snapshot under ``path`` into a :class:`SnapshotCache`.

    This is the pipeline's one expensive step, and the only place a plotting run
    reads snapshot data.  It parallelizes over snapshots because they are
    completely independent.

    Parameters
    ----------
    path
        A snapshot directory, or a single ``.h5`` file.
    points
        Reconstruction resolution per axis.  Cost scales as ``points**2``.
    workers
        Pool width; ``None``/0 uses every logical CPU.
    dt
        Solver time step, used only to infer times for snapshots that predate
        the stored time field.
    warn_bytes
        Print a warning once the cache exceeds this size.  The default (2 GB) is
        well under a typical allocation but high enough not to fire in normal
        use; it exists so an unreasonable ``--points`` reports itself here
        instead of as an OOM kill several minutes later.

    Frames come back in ascending *time* order, not filename order, because
    everything downstream (movies, growth-rate differencing, time series)
    assumes monotonic time.
    """
    files = snapshot_files(path)
    tasks = [
        {"index": index, "path": file_path, "points": points, "dt": dt}
        for index, file_path in enumerate(files)
    ]

    print(
        f"reading {len(files)} snapshots at {points}x{points} "
        f"with {min(worker_count(workers), max(1, len(files)))} workers",
        flush=True,
    )

    results = [None] * len(files)
    pool_width = min(worker_count(workers), max(1, len(tasks)))
    if pool_width == 1 or len(tasks) == 1:
        # Serial path keeps single-core runs debuggable and skips pool startup.
        for task in tasks:
            index, values, x_axis, y_axis, time = _snapshot_load_task(task)
            results[index] = (values, x_axis, y_axis, time)
    else:
        with process_pool(pool_width) as pool:
            # chunksize=1: tasks are expensive and their cost varies with how
            # refined each snapshot's grid is, so larger chunks would strand
            # work at the tail.
            for index, values, x_axis, y_axis, time in pool.map(
                _snapshot_load_task, tasks, chunksize=1
            ):
                results[index] = (values, x_axis, y_axis, time)

    # Sort by time.  Filename order is usually already time order, but a
    # restarted run can break that, and every consumer assumes monotonicity.
    order = sorted(range(len(results)), key=lambda i: results[i][3])
    cache = SnapshotCache(
        frames=[results[i][0] for i in order],
        times=[results[i][3] for i in order],
        # Axes are identical across frames, so keep one copy.
        x=results[order[0]][1],
        y=results[order[0]][2],
        files=[files[i] for i in order],
        points=points,
    )

    total = cache.nbytes
    print(f"snapshot cache: {total / 1e6:.1f} MB for {len(cache)} frames",
          flush=True)
    if total > warn_bytes:
        print(
            f"warning: the snapshot cache is {total / 1e9:.1f} GB. Reduce "
            f"--points or plot fewer snapshots if this approaches the job's "
            f"memory limit.",
            flush=True,
        )
    return cache


# ---------------------------------------------------------------------------
# Section 5: numerical noise floor
# ---------------------------------------------------------------------------
#
# Lives here rather than in calculations.py because what it does is *read* the
# solver's own tolerances out of the input deck -- it is a deck query, not a
# derived numerical quantity.

# Machine epsilon is the smallest relative difference float64 can represent.
DEFAULT_NUMERICAL_TOLERANCE = np.finfo(float).eps
# Ten times epsilon: a conservative "nothing below this is real" floor for
# when the deck specifies no solver tolerance at all.
DEFAULT_MACHINE_DISPLAY_FLOOR = 10.0 * DEFAULT_NUMERICAL_TOLERANCE

# Deck keys whose values bound what is numerically trustworthy.  Both
# spellings (with and without the leading dash) appear across deck
# generations, and both the linear solver tolerance and the adaptivity
# threshold matter -- whichever is looser sets the real floor.
_TOLERANCE_KEYS = ("-isolve-tol", "isolve-tol", "-adapt", "adapt")


def numerical_display_floor(solver_input):
    """Return the value below which solver output is not trustworthy.

    Reads the solver's own tolerances out of the input deck rather than
    hardcoding a number, so the floor automatically tracks how the run was
    configured.

    Why this matters: signed-log plots must not render numerical noise as
    physically meaningful structure.  Below the solver's convergence
    tolerance, the sign and magnitude of ``f`` are artifacts of the linear
    solve, not physics -- so plotting them produces confident-looking
    garbage.

    Takes the **maximum** of the available tolerances (loosest wins), since a
    quantity is only trustworthy if it clears *every* source of numerical
    error.  Falls back to ten times machine epsilon when the deck names no
    tolerance.

    Individual key parses are wrapped in ``try`` because a malformed or empty
    value should be skipped, not crash the whole plotting run.
    """
    options = read_options(solver_input)
    tolerances = []
    for key in _TOLERANCE_KEYS:
        if key not in options:
            continue
        try:
            # Values may be vectors ("-adapt : 1e-6 1e-6"); the first token
            # is the relevant scalar.
            value = float(options[key].split()[0])
        except (TypeError, ValueError, IndexError):
            continue
        # Reject non-finite and non-positive values: a zero or negative
        # "tolerance" is a deck error, and using it would disable the floor.
        if np.isfinite(value) and value > 0.0:
            tolerances.append(value)
    if tolerances:
        return max(tolerances)
    return DEFAULT_MACHINE_DISPLAY_FLOOR

# ---------------------------------------------------------------------------
# Section 5b: the 1-D snapshot cache (ICRF_1D / LHCD_1D)
# ---------------------------------------------------------------------------
#
# Identical idea to SnapshotCache/load_snapshots, one dimension down: the 1-D
# projects reconstruct curves with plot_data1d instead of images with
# plot_data2d, so frames are (points,) vectors sharing one coordinate axis.


def _snapshot_load_task_1d(task):
    """Worker: reconstruct one 1-D snapshot; see :func:`_snapshot_load_task`."""
    snapshot = asgard.pde_snapshot(task["path"])
    values, axis = snapshot.plot_data1d(((),), num_points=task["points"])
    time = snapshot_time(task["path"])
    if time is None:
        step = snapshot_step(task["path"])
        if step is None:
            step = task["index"]
        time = step * task["dt"]
    return (task["index"], np.asarray(values, dtype=np.float32),
            np.asarray(axis, dtype=float), float(time))


@dataclass
class SnapshotCache1D:
    """Every reconstructed 1-D snapshot of one run: the cache the 1-D
    projects' plotters consume.  Same contract as :class:`SnapshotCache`
    with ``y`` absent and frames as ``(points,)`` vectors."""

    frames: list
    times: list
    x: object
    files: list
    points: int

    def __len__(self):
        return len(self.frames)


def load_snapshots_1d(path, points, workers=None, dt=1.0):
    """Read every 1-D snapshot under ``path`` into a :class:`SnapshotCache1D`.

    The 1-D analogue of :func:`load_snapshots`; cheap enough that no memory
    warning is needed (a frame is a vector, not an image).
    """
    files = snapshot_files(path)
    tasks = [
        {"index": index, "path": file_path, "points": points, "dt": dt}
        for index, file_path in enumerate(files)
    ]

    pool_width = min(worker_count(workers), max(1, len(tasks)))
    print(f"reading {len(files)} 1-D snapshots at {points} points "
          f"with {pool_width} workers", flush=True)

    results = [None] * len(files)
    if pool_width == 1 or len(tasks) == 1:
        for task in tasks:
            index, values, axis, time = _snapshot_load_task_1d(task)
            results[index] = (values, axis, time)
    else:
        with process_pool(pool_width) as pool:
            for index, values, axis, time in pool.map(
                _snapshot_load_task_1d, tasks, chunksize=4
            ):
                results[index] = (values, axis, time)

    order = sorted(range(len(results)), key=lambda i: results[i][2])
    return SnapshotCache1D(
        frames=[results[i][0] for i in order],
        times=[results[i][2] for i in order],
        x=results[order[0]][1],
        files=[files[i] for i in order],
        points=points,
    )
