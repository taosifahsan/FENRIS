#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
./nersc_login_once.sh

echo
echo "Done. You can close this window."
