#!/usr/bin/env bash
if [[ -d /root/miniconda3/bin ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_CALIBRATION_OUTPUT="${LEMAN_CALIBRATION_OUTPUT:-$LEMAN_GP_ROOT/output/leman_gazetteer_v6_seed42}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$LEMAN_CALIBRATION_OUTPUT"
