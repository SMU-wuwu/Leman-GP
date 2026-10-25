#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Default is dev. Test is evaluated only when explicitly passing --split test.
torchrun --standalone --nnodes=1 --nproc_per_node=4 predict_calibrated.py "$@" 2>&1 | tee -a "$LEMAN_CALIBRATION_OUTPUT/predict.log"
python collect_results.py
