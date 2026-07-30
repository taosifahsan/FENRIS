#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
./run_nersc_and_fetch.sh

echo
echo "Done. You can close this window."
