#!/usr/bin/env bash
# ICRF_1D: build + solve + plot.  Edit the parameters below and run:
#
#   tools/run.sh
#
# Load your environment first (venv, modules); this assumes cmake, the
# compilers, and the ASGarD python are already on PATH.  With the venv active,
# find_package(asgard) locates the package by itself; export asgard_DIR only
# if it lives somewhere unusual.
#
# Staleness is CMake's: the solve re-runs only when its inputs changed; the
# plot stage always renders.  To force a re-solve, delete the
# output_data/movie directory (or just its .solve-complete sentinel).

set -euo pipefail

# ----- cores ----------------------------------------------------------------
BUILD_CORES=8       # compile parallelism
PLOT_CORES=0        # plotter worker processes (0 = all cores)

# ----- endpoint:  plots (solve if stale, then figures) | run (solve only) ---
TARGET=plots

# -----------------------------------------------------------------------------
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -S "${project_root}" -B "${project_root}/build" \
    -DICRF_1D_PLOT_WORKERS="${PLOT_CORES}"

cmake --build "${project_root}/build" -j "${BUILD_CORES}" --target "${TARGET}"
