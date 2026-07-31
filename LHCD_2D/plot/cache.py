"""Build the shared snapshot cache: the single expensive read of a plot run.

Stage one of ``tools/run.sh``: reconstructing snapshots through asgard costs
seconds, so it happens exactly once, here, and the result is saved to one
``.npz`` file.  The plotters that run.sh then launches in parallel load that
file back in milliseconds via their ``--cache`` flag instead of each
re-reading the snapshots.

Used by: ``tools/run.sh`` (stage one, before the parallel plotters launch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Locate the directory that holds plot_common by walking up from this file,
# so reorganizing the project tree cannot silently break the import.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "plot_common").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("plot_common not found above " + __file__)

from plot_common.runtime import bootstrap

PATHS = bootstrap(__file__)

from plot_common.reader import load_snapshots, save_cache


def main():
    """CLI entry point: reconstruct every snapshot once, write the .npz cache."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=str(PATHS.snapshots))
    parser.add_argument("-n", "--points", type=int, default=192)
    parser.add_argument("--out", default=str(PATHS.snapshots.parent / "cache.npz"))
    args = parser.parse_args()
    save_cache(load_snapshots(args.output, args.points), args.out)


if __name__ == "__main__":
    main()
