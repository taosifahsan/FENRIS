// The initial distribution f0(x, theta), written as code -- this file is an
// input.
//
// Edit the lambda body below and rerun tools/run.sh: the header is a tracked
// compile input of the solver, so the build recompiles and the solve reruns
// automatically.  No other file needs to change.
//
// Contract:
//   - x = v/v_ta (speed), spanning 0..x_max from the deck; theta is the pitch
//     angle in radians, 0..pi.  The function is fully 2-D: any pitch
//     dependence is allowed, not just separable products.
//   - coll_eq is the zero-flux collisional equilibrium shape at this x,
//     computed from the coefficient tables -- the physical pre-RF start.
//     The default simply returns it; replace the body to start from anything
//     else (and ignore coll_eq freely).
//   - Return any non-negative shape.  Do NOT normalize and do NOT include the
//     evolution measure x^2 lambda(theta) sin(theta): the solver applies the
//     measure itself and divides by
//     2 pi * integral f0 * x^2 lambda(theta) sin(theta) dx dtheta
//     (the theta integral resolving the trapped-passing peak of lambda), so
//     the full 3-D particle count -- gyrophase 2 pi included -- always
//     starts at exactly 1, matching LHCD_2D's convention.
//
// Used by: src/ICRF_2D.cpp (compiled in via #include).  A copy is archived
// into each run's figures/<timestamp>/ next to the deck.

#pragma once

#include <cmath>

inline const auto initial_f0 = [](double x, double theta,
                                  double coll_eq) -> double {
    (void)x; (void)theta;
    // The pre-RF collisional equilibrium, pitch-independent.
    return coll_eq;
};
