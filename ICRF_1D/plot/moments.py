"""Particle conservation, energy, and power for ICRF_1D.

The implementation lives in :mod:`plot_common.moments_methods`, shared by all
four projects.  This file supplies the one thing that genuinely differs --
the phase-space measure (the Jacobian) -- plus the figure labels.

Measure: ``N(t) = int f v^2 dv``;  ``E(t) = int m v^2 f v^2 dv``, with the minority
mass ``m`` (amu) from the deck -- the isotropic-speed Jacobian ``v^2``.
With the standard ``v = v_phys/sqrt(2T_e/m_0)`` a particle's kinetic energy
is exactly ``m T_e v^2``, so E/(N T_e) starts at the equipartition
``(3/2) m T_minority/T_e`` -- 3/2 when the minority sits at the electron
temperature.

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
from plot_common.reader import option_float

if __name__ == "__main__":
    main_1d(
        PATHS,
        description="ICRF_1D moment diagnostics",
        number_weight=lambda v, options: v**2,
        energy_weight=lambda v, options: option_float(options, "m") * v**4,
    )
