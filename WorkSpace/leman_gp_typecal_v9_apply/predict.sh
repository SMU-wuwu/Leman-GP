#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/root/miniconda3/bin:$PATH"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_CALIBRATION_OUTPUT="${LEMAN_CALIBRATION_OUTPUT:-$LEMAN_GP_ROOT/output/leman_typecal_v9_seed42}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Default dev; test only on an explicit future user request.
torchrun --standalone --nnodes=1 --nproc_per_node=4 predict.py "$@" 2>&1 | tee -a "$LEMAN_CALIBRATION_OUTPUT/predict.log"
