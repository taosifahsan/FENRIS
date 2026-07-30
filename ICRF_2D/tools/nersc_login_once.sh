#!/usr/bin/env bash
set -euo pipefail

nersc_user="${NERSC_USER:-taosif}"
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v sshproxy >/dev/null 2>&1; then
    sshproxy_bin="$(command -v sshproxy)"
elif [ -x "${tool_dir}/sshproxy" ]; then
    sshproxy_bin="${tool_dir}/sshproxy"
else
    cat <<EOF
sshproxy is not installed on this Mac.

Install the macOS package or download sshproxy from:
    https://portal.nersc.gov/cfs/mfa/

After installing it, run:
    tools/nersc_login_once.sh

EOF
    exit 1
fi

"${sshproxy_bin}" -u "${nersc_user}"

echo
echo "NERSC temporary SSH key:"
ssh-keygen -L -f "${HOME}/.ssh/nersc-cert.pub" | grep Valid || true
