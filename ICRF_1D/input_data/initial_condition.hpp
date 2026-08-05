// The initial distribution f0(v), written as code -- this file is an input.
//
// Edit the lambda body below and rerun tools/run.sh: the header is a tracked
// compile input of the solver, so the build recompiles and the solve reruns
// automatically.  No other file needs to change.
//
// Contract:
//   - v is the speed in units of the electron thermal velocity (deck domain).
//   - m is the deck's minority mass parameter (the same "m" input_solver.txt
//     sets), passed in so the default can follow the deck; ignore it freely.
//   - Return any non-negative shape.  Do NOT normalize: the solver divides
//     by integral f0(v) v^2 dv over the domain, so the initial state always
//     carries exactly unit particle number regardless of prefactors.
//
// Used by: src/ICRF_1D.cpp (compiled in via #include).  A copy is archived
// into each run's figures/<timestamp>/ next to the deck.

#pragma once

#include <cmath>

inline const auto initial_f0 = [](double v, double m) -> double {
    // Maxwellian at the background temperature (the pre-RF equilibrium).
    return std::exp(-m * v * v);
};
