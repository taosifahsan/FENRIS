"""The adaptive sparse grid for LHCD_1D: refinement level and DOF count.

The implementation lives in :mod:`plot_common.grid_methods`, shared by all four
projects -- these four files were 96-99% identical, differing only in the
axis labels and the argparse description supplied below.  See that module
for what the two figures are and how the refinement geometry is computed.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.grid_methods`.
"""

from __future__ import annotations

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

from plot_common.grid_methods import main_1d


if __name__ == "__main__":
    main_1d(PATHS, description="LHCD_1D adaptive-grid plots",
             xlabel=r"$v_{\parallel}$")
