#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LEMAN_GP_SEED=44 LEMAN_GP_EPOCHS=3 LEMAN_GP_BATCH_SIZE=32
export LEMAN_GP_ENCODER_LR=0.000003
export LEMAN_GP_REUSED_HEAD_LR=0.00001 LEMAN_GP_HEAD_LR=0.00001
export LEMAN_GP_OUTPUT="$LEMAN_GP_ROOT/output/leman_fgm_v14_seed44"
export LEMAN_GP_WARM_START="$LEMAN_GP_ROOT/output/leman_finetune_v4_seed42/best.pt"
"${LEMAN_GP_PYTHON:-/root/miniconda3/bin/python}" prepare.py
exec bash "$LEMAN_GP_ROOT/leman_gp_fgm_v10/train.sh" "$@"
