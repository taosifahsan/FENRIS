#!/usr/bin/env bash
if [ -n "${OMP_NUM_THREADS:-}" ]; then
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    exec "$@"
fi

cpu_count="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-${PBS_NP:-${NSLOTS:-}}}}"
if [ -z "$cpu_count" ]; then
    cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
    if [ 0 -gt 0 ] && [ "$cpu_count" -gt 0 ]; then
        cpu_count=0
    fi
fi
case "$cpu_count" in
    ''|*[!0-9]*) cpu_count=1 ;;
esac
export OMP_NUM_THREADS="$cpu_count"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
exec "$@"
