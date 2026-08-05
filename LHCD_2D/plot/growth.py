"""The fractional growth rate for LHCD_2D: ``gamma_f = d(ln f)/dt``.

The implementation lives in :mod:`plot_common.growth_methods`, shared by all four
projects -- this piece really is identical across them.  This file supplies
only the initial-condition shape and the labels.  Particle conservation and
energy/power live in this folder's ``moments.py``, since those measures
genuinely differ per project.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.growth_methods`.
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

import numpy as np

from coefficients import style_cartesian_axes
from plot_common.growth_methods import main_2d

if __name__ == "__main__":
    main_2d(
        PATHS,
        description="LHCD growth rate",
        symbol=r"f",
        style_axes=style_cartesian_axes,
        # initial_condition.hpp: exp(-x*x) in the standard sqrt(2T/m)
        # units.  Only the SHAPE matters -- the amplitude is calibrated
        # against the solver's own first frame.
        initial_shape=lambda cache: np.exp(
            -np.asarray(cache.x, dtype=float) ** 2),
    )
