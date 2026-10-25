#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
export LEMAN_REFINE_OUTPUT="${LEMAN_REFINE_OUTPUT:-$LEMAN_GP_ROOT/output/leman_decode_v3_seed$LEMAN_GP_SEED}"
exec 9>"$LEMAN_REFINE_OUTPUT/.decode.lock"
flock -n 9 || { echo 'This decoding experiment is already running.'; exit 1; }
torchrun --standalone --nnodes=1 --nproc_per_node=4 refine.py test 2>&1 | tee -a "$LEMAN_REFINE_OUTPUT/test.log"
python pack_results.py
