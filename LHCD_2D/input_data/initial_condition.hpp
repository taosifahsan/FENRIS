// The initial distribution f0(x, theta), written as code -- this file is an
// input.
//
// Edit the lambda body below and rerun tools/run.sh: the header is a tracked
// compile input of the solver, so the build recompiles and the solve reruns
// automatically.  No other file needs to change.
//
// Contract:
//   - x = v/v_th with v_th = sqrt(2 T_e/m_e) -- the convention standardized
//     across all four projects; see the normalization header in
//     src/LHCD_2D.cpp -- spanning 0..x_max from the deck; theta is the pitch
//     angle in radians, 0..pi.  The function is fully 2-D: any pitch
//     dependence is allowed, not just separable products.
//   - Return any non-negative shape.  Do NOT normalize and do NOT include the
//     phase-space Jacobian x^2 sin(theta): the solver applies the Jacobian
//     itself and divides by the 3-D velocity-space norm
//     2 pi * integral f0 x^2 sin(theta) dx dtheta,
//     so the default Maxwellian below reproduces the textbook
//     pi^{-3/2} exp(-x^2) exactly.
//
// Used by: src/LHCD_2D.cpp (compiled in via #include).  A copy is archived
// into each run's figures/<timestamp>/ next to the deck.

#pragma once

#include <cmath>

inline const auto initial_f0 = [](double x, double theta) -> double {
    (void)theta;   // isotropic default
    // Maxwellian at the background temperature (x in units of sqrt(2T/m)).
    return std::exp(-x * x);
};
