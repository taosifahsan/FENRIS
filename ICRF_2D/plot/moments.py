"""Particle conservation, energy, and power for ICRF_2D.

The implementation lives in :mod:`plot_common.moments_methods`, shared by all
four projects.  This file supplies the one thing that genuinely differs --
the phase-space measure (the Jacobian) -- plus the figure labels.

Measure: the solver's phase-space measure, whose theta Jacobian is the
bounce-orbit weight ``lambda(theta_0)`` from the coefficient table rather
than a plain ``sin(theta)``.  Energy is in units of T_a: since
``x = v/v_ta`` and ``v_ta^2 = 2 T_a/m_a``, one particle's kinetic energy is
exactly ``T_a x^2``, so E/(N T_a) starts at the equipartition 3/2 (multiply
by the table's T_a for keV).

Used by: ``tools/run.sh`` (one of the parallel plotter processes).

Depends on: :mod:`plot_common.moments_methods`, ``coefficients.py`` (the bounce-orbit
quadrature and species temperature).
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

from coefficients import bounce_pitch_quadrature
from plot_common.moments_methods import main_2d

if __name__ == "__main__":
    main_2d(
        PATHS,
        description="ICRF moment diagnostics",
        jacobian_th=bounce_pitch_quadrature,
        energy_title=r"Bounce-weighted fast-ion energy",
        energy_ylabel=r"$E_\lambda/(N_\lambda(0)\,T_a)$",
        power_title=r"Net fast-ion power, $P_\lambda=dE_\lambda/dt$",
        power_ylabel=r"$P_\lambda/(N_\lambda(0)\,T_a)$ [$\tau_c^{-1}$]",
    )
