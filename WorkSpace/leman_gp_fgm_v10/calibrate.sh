#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
exec 9>"$LEMAN_GP_OUTPUT/.training.lock"
flock -n 9 || { echo 'This output directory is already in use.'; exit 1; }
torchrun --standalone --nnodes=1 --nproc_per_node=4 calibrate.py "$@" 2>&1 | tee -a "$LEMAN_GP_OUTPUT/calibrate.log"
