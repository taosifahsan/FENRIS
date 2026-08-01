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
# Staleness is CMake's: the solve re-runs only when its inputs changed.
# To force a re-solve, delete the output_data/movie directory (or just its
# .solve-complete sentinel).
#
# Plotting always runs, in two stages:
#   1. plot/cache.py reads every snapshot ONCE and saves the reconstructed
#      cache to output_data/cache.npz (the only expensive read).
#   2. every plotter runs in parallel, loading that cache in milliseconds.
# Everything lands in one fresh timestamped directory under figures/, with
# the input deck(s) copied alongside and the full console record -- solver
# prints, [time] stage timings, every saved figure -- in run.out.

set -euo pipefail

# ----- cores ----------------------------------------------------------------
BUILD_CORES=8       # compile parallelism
SOLVER_CORES=4      # ASGarD solve OpenMP threads (4 measured fastest, ~1.2x
                    # over the old 2; >10 spills onto E-cores, much slower)

# -----------------------------------------------------------------------------
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

figs="figures/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${figs}"

# timed CMD ARGS...: run the command, then print "[time] <arg2>: N s".
timed() { local s=$SECONDS; "$@"; local rc=$?; echo "[time] ${2}: $((SECONDS - s)) s"; return $rc; }

{
t_total=$SECONDS

cmake -S "${project_root}" -B "${project_root}/build" \
    -DICRF_1D_SOLVER_OMP_THREADS="${SOLVER_CORES}"

t=$SECONDS
cmake --build "${project_root}/build" -j "${BUILD_CORES}" --target run
echo "[time] build + solve: $((SECONDS - t)) s"

# ----- plots: one shared read, then every plotter in parallel ---------------
cache="output_data/cache.npz"
# the cache is a scratch artifact of this run: delete it on exit, success or
# failure (the plotters have already loaded it into memory by then).
trap 'rm -f "${cache}"' EXIT

timed python3 plot/cache.py --out "${cache}"

t=$SECONDS
pids=()
timed python3 plot/solution.py    --cache "${cache}" --fig-dir "${figs}" & pids+=($!)
timed python3 plot/temperature.py --cache "${cache}" --fig-dir "${figs}" & pids+=($!)
timed python3 plot/diagnostics.py --cache "${cache}" --fig-dir "${figs}" & pids+=($!)
timed python3 plot/grid.py        --fig-dir "${figs}" & pids+=($!)
# wait on each PID: a bare `wait` ignores job failures, this aborts on one
for pid in "${pids[@]}"; do wait "${pid}"; done
echo "[time] plotters, parallel wall: $((SECONDS - t)) s"

cp input_data/input_solver.txt "${figs}/"
echo "[time] total: $((SECONDS - t_total)) s"
echo "saved everything in ${figs}"
} 2>&1 | tee "${figs}/run.out"
