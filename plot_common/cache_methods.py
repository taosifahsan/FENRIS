"""Build the shared snapshot cache: the single expensive read of a plot run.

Stage one of ``tools/run.sh``: reconstructing snapshots through asgard costs
seconds, so it happens exactly once, here, and the result is saved to one
``.npz`` file.  The plotters that run.sh then launches in parallel load that
file back in milliseconds via their ``--cache`` flag instead of each
re-reading the snapshots.

The fourth of the shared method modules, alongside
:mod:`plot_common.grid_methods`, :mod:`plot_common.growth_methods` and
:mod:`plot_common.moments_methods`.  Nothing project-specific survives here
at all: the four ``plot/cache.py`` files were byte-identical within each
dimensionality, and 1-D differs from 2-D only in which loader it calls and
how the default reconstruction resolution is chosen.  So the wrappers pass
nothing but their own ``PATHS``.

Used by: each project's ``plot/cache.py``, in turn by ``tools/run.sh``
(stage one, before the parallel plotters launch).

Depends on: :mod:`plot_common.reader` (the loaders, ``save_cache``).
"""

from __future__ import annotations

import argparse

from plot_common.reader import (
    load_snapshots,
    load_snapshots_1d,
    option_float,
    read_options,
    save_cache,
)


def _parse(paths, default_points, points_help=None):
    """The two-flag surface, plus ``--out`` for the cache file itself."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=str(paths.snapshots))
    parser.add_argument("-n", "--points", type=int, default=default_points,
                        help=points_help)
    parser.add_argument("--out",
                        default=str(paths.snapshots.parent / "cache.npz"))
    return parser.parse_args()


def main_1d(paths):
    """CLI entry point for a 1-D project's ``plot/cache.py``.

    ``-n 0`` (the default) takes the resolution from the deck's
    ``num_points``, halved -- the reconstruction is already smooth at that
    density, and asking for more only costs time.
    """
    args = _parse(paths, default_points=0,
                  points_help="reconstruction points (0 = deck num_points / 2)")
    points = args.points
    if points <= 0:
        points = int(option_float(read_options(paths.solver_input),
                                  "num_points", 256) / 2)
    save_cache(load_snapshots_1d(args.output, points), args.out)


def main_2d(paths):
    """CLI entry point for a 2-D project's ``plot/cache.py``."""
    args = _parse(paths, default_points=192)
    save_cache(load_snapshots(args.output, args.points), args.out)
