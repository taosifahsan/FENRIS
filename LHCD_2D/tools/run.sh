#!/usr/bin/env bash
# LHCD_2D: build + solve + plot.  Edit the parameters below and run:
#
#   tools/run.sh
#
# Load your environment first (venv, modules); this assumes cmake, the
# compilers, and the ASGarD python are already on PATH.  With the venv active,
# find_package(asgard) locates the package by itself; export asgard_DIR only
# if it lives somewhere unusual.
#
# Staleness is CMake's: tables and the solve re-run only when their inputs
# changed; the plot stage always renders.  To force a re-solve, delete
# output_data/output.h5.

set -euo pipefail

# ----- cores ----------------------------------------------------------------
BUILD_CORES=8       # compile parallelism
SOLVER_CORES=2      # ASGarD solve OpenMP threads (2 measured fastest)
PLOT_CORES=0        # plotter worker processes (0 = all cores)

# ----- endpoint:  plots (solve if stale, then figures) | run (solve only) ---
TARGET=plots

# -----------------------------------------------------------------------------
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cmake -S "${project_root}" -B "${project_root}/build" \
    -DLHCD_SOLVER_OMP_THREADS="${SOLVER_CORES}" \
    -DLHCD_PLOT_WORKERS="${PLOT_CORES}"

cmake --build "${project_root}/build" -j "${BUILD_CORES}" --target "${TARGET}"
