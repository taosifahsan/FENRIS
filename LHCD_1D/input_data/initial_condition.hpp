// The initial distribution f0(v), written as code -- this file is an input.
//
// Edit the lambda body below and rerun tools/run.sh: the header is a tracked
// compile input of the solver, so the build recompiles and the solve reruns
// automatically.  No other file needs to change.
//
// Contract:
//   - v is the parallel velocity, spanning the deck's domain_min..domain_max.
//   - Return any non-negative shape.  Do NOT normalize: the solver divides
//     by integral f0(v) dv over the domain (the parallel-velocity particle
//     measure), so the initial state always carries exactly unit particle
//     number regardless of prefactors.  (Diagnostics divide by N(0) and the
//     plotters renormalize, so only the shape matters downstream.)
//
// Used by: src/LHCD_1D.cpp (compiled in via #include).  A copy is archived
// into each run's figures/<timestamp>/ next to the deck.

#pragma once

#include <cmath>

inline const auto initial_f0 = [](double v) -> double {
    // Maxwellian in parallel velocity.
    return std::exp(-v * v);
};
