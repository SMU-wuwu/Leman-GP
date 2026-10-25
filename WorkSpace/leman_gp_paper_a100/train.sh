#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_GP_SEED="${LEMAN_GP_SEED:-42}"
export LEMAN_GP_OUTPUT="${LEMAN_GP_OUTPUT:-$LEMAN_GP_ROOT/output/leman_paper_a100_seed$LEMAN_GP_SEED}"
export PYTHONHASHSEED="$LEMAN_GP_SEED"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LEMAN_GP_OUTPUT"
exec 9>"$LEMAN_GP_OUTPUT/.training.lock"
flock -n 9 || { echo 'This output directory is already being trained.'; exit 1; }
if [ -f "$LEMAN_GP_OUTPUT/STOP" ]; then
    echo "Remove $LEMAN_GP_OUTPUT/STOP before resuming."
    exit 1
fi
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py "$@" 2>&1 | tee -a "$LEMAN_GP_OUTPUT/train.log"
