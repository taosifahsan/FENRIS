"""Particle conservation, energy, and power for LHCD_2D.

The implementation lives in :mod:`plot_common.moments_methods`, shared by all
four projects.  This file supplies the one thing that genuinely differs --
the phase-space measure (the Jacobian) -- plus the figure labels.

Measure: the solver's ``x^2 sin(theta)``.  LHCD has no orbit geometry, so
the theta Jacobian is the closed-form ``sin(theta)``, folded into the
trapezoid quadrature here.  Energy is in units of T_e: with the standard
``x = v/sqrt(2T_e/m_e)`` a particle's kinetic energy is exactly ``T_e x^2``,
so E/(N T_e) starts at the equipartition 3/2.

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

from plot_common.moments_methods import main_2d, trapezoid_weights

if __name__ == "__main__":
    main_2d(
        PATHS,
        description="LHCD moment diagnostics",
        jacobian_th=lambda theta: trapezoid_weights(theta) * np.sin(theta),
        energy_title=r"Fast-electron energy",
        energy_ylabel=r"$E/(N(0)\,T_e)$",
        power_title=r"Net fast-electron power, $P=dE/dt$",
        power_ylabel=r"$P/(N(0)\,T_e)$ [$\tau_c^{-1}$]",
    )
