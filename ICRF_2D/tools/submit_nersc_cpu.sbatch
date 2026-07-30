#!/usr/bin/env bash
#SBATCH -J icrf_2d
#SBATCH -A m77
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs output_data figures

if command -v module >/dev/null 2>&1; then
    module load cmake
fi

threads="${SLURM_CPUS_PER_TASK:-}"
if [ -z "${threads}" ]; then
    threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
export OMP_NUM_THREADS="${threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

build_dir="${ICRF_BUILD_DIR:-build_nersc}"
cmake_args=(-S . -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release)
if [ -n "${asgard_DIR:-}" ]; then
    cmake_args+=("-Dasgard_DIR=${asgard_DIR}")
elif [ -f "${HOME}/.conda/envs/asgard/asgard-config.cmake" ]; then
    cmake_args+=("-Dasgard_DIR=${HOME}/.conda/envs/asgard")
elif [ -n "${asgard_ROOT:-}" ]; then
    cmake_args+=("-Dasgard_ROOT=${asgard_ROOT}")
elif [ -n "${ASGARD_ROOT:-}" ]; then
    cmake_args+=("-Dasgard_ROOT=${ASGARD_ROOT}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target run
