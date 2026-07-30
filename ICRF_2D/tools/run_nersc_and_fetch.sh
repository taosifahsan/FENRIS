#!/usr/bin/env bash
set -euo pipefail

nersc_user="${NERSC_USER:-taosif}"
nersc_project_dir="${NERSC_PROJECT_DIR:-/pscratch/sd/t/taosif/ICRF_runs/ICRF_2D}"

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${tool_dir}/.." && pwd)"

ssh_transport="ssh"
ssh_cmd=(ssh)
if [ -f "${HOME}/.ssh/nersc" ]; then
    ssh_transport="ssh -i ${HOME}/.ssh/nersc -o IdentitiesOnly=yes"
    ssh_cmd=(ssh -i "${HOME}/.ssh/nersc" -o IdentitiesOnly=yes)
fi

echo "Syncing project to NERSC..."
echo "    ${project_dir}/"
echo " -> ${nersc_user}@dtn.nersc.gov:${nersc_project_dir}/"
echo

rsync -av \
    -e "${ssh_transport}" \
    --exclude build \
    --exclude build_nersc \
    --exclude .DS_Store \
    "${project_dir}/" \
    "${nersc_user}@dtn.nersc.gov:${nersc_project_dir}/"

echo
echo "Submitting on Perlmutter and waiting for the Slurm job to finish..."
echo

"${ssh_cmd[@]}" "${nersc_user}@perlmutter.nersc.gov" \
    "cd '${nersc_project_dir}' && job_id=\$(sbatch --parsable tools/submit_nersc_cpu.sbatch) && echo \"Submitted job \${job_id}\" && while [ -n \"\$(squeue -h -j \"\${job_id}\")\" ]; do date; squeue -j \"\${job_id}\"; sleep 60; done && echo \"Job \${job_id} left the queue\" && sacct -j \"\${job_id}\" --format=JobID,JobName,State,Elapsed,ExitCode"

echo
echo "Fetching finished results..."
echo

mkdir -p "${project_dir}/output_data" "${project_dir}/figures" "${project_dir}/logs"

rsync -av \
    -e "${ssh_transport}" \
    --include='output_data/***' \
    --include='figures/***' \
    --include='logs/***' \
    --include='icrf_2d-*.out' \
    --include='slurm-*.out' \
    --exclude='*' \
    "${nersc_user}@dtn.nersc.gov:${nersc_project_dir}/" \
    "${project_dir}/"

if compgen -G "${project_dir}/icrf_2d-*.out" >/dev/null; then
    mv "${project_dir}"/icrf_2d-*.out "${project_dir}/logs/" 2>/dev/null || true
fi
if compgen -G "${project_dir}/slurm-*.out" >/dev/null; then
    mv "${project_dir}"/slurm-*.out "${project_dir}/logs/" 2>/dev/null || true
fi

echo
echo "Done. Results are in:"
echo "    ${project_dir}/output_data"
echo "    ${project_dir}/figures"
echo "    ${project_dir}/logs"
