"""Build the shared snapshot cache for LHCD_1D: the one expensive read.

The implementation lives in :mod:`plot_common.cache_methods`, shared by all
four projects.  Nothing here is project-specific -- these files were
byte-identical within each dimensionality -- so this wrapper passes only its
own ``PATHS``.

Used by: ``tools/run.sh`` (stage one, before the parallel plotters launch).

Depends on: :mod:`plot_common.cache_methods`.
"""

from __future__ import annotations

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

from plot_common.cache_methods import main_1d


if __name__ == "__main__":
    main_1d(PATHS)
