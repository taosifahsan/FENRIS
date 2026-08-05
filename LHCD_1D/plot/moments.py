"""Particle conservation, energy, and power for LHCD_1D.

The implementation lives in :mod:`plot_common.moments_methods`, shared by all
four projects.  This file supplies the one thing that genuinely differs --
the phase-space measure (the Jacobian) -- plus the figure labels.

Measure: ``N(t) = int f dv``;  ``E(t) = int v^2 f dv`` -- no theta Jacobian, since
the 1-D parallel equation carries none.  With the standard
``v = v_phys/sqrt(2T/m)`` a particle's kinetic energy is exactly ``T v^2``,
so E/(N T) starts at the 1-D equipartition value 1/2.

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.moments_methods`.
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

from plot_common.moments_methods import main_1d

if __name__ == "__main__":
    main_1d(
        PATHS,
        description="LHCD_1D moment diagnostics",
        number_weight=lambda v, options: np.ones_like(v),
        energy_weight=lambda v, options: v**2,
    )
