#!/usr/bin/env bash
#SBATCH --job-name=lhcd_2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#
# Minimal batch launcher.  Submit FROM THE PROJECT ROOT:
#
#   sbatch tools/submit.sh
#
# Environment setup lives here; the pipeline and its core-count knobs live in
# tools/run.sh -- edit that file, not this one, to change how the run behaves.

set -euo pipefail

module purge
module load community-modules
module load gcc/12.2.0 cmake/3.27.9 openblas/0.3.26
module load nvhpc/24.5 openmpi/4.1.7 hdf5/1.14.3
module load ffmpeg 2>/dev/null || true    # optional: stills still save without it

source "${HOME}/venv_asgard/bin/activate"
# Not strictly required -- find_package(asgard) discovers the active venv via
# PATH -- but it protects against a stray unconfigured asgard-config.cmake at
# the venv root (a known ASGarD install artifact), which would otherwise
# shadow the real config.  Delete once ~/venv_asgard is confirmed clean.
export asgard_DIR="${VIRTUAL_ENV}/lib/cmake/asgard"

# SLURM_SUBMIT_DIR because Slurm copies this script to a spool directory,
# so its own path says nothing about where the project is.
bash "${SLURM_SUBMIT_DIR:-.}/tools/run.sh"
