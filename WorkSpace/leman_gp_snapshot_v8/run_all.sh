#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/root/miniconda3/bin:$PATH"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_CALIBRATION_OUTPUT="${LEMAN_CALIBRATION_OUTPUT:-$LEMAN_GP_ROOT/output/leman_snapshot_v8_seed42}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LEMAN_CALIBRATION_OUTPUT"
torchrun --standalone --nnodes=1 --nproc_per_node=4 extract.py 2>&1 | tee "$LEMAN_CALIBRATION_OUTPUT/extract.log"
python calibrate.py 2>&1 | tee "$LEMAN_CALIBRATION_OUTPUT/calibrate.log"
