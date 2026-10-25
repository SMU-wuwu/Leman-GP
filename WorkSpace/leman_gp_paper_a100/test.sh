#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_GP_SEED="${LEMAN_GP_SEED:-42}"
export LEMAN_GP_OUTPUT="${LEMAN_GP_OUTPUT:-$LEMAN_GP_ROOT/output/leman_paper_a100_seed$LEMAN_GP_SEED}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LEMAN_GP_OUTPUT"
torchrun --standalone --nnodes=1 --nproc_per_node=2 test.py "$@" 2>&1 | tee -a "$LEMAN_GP_OUTPUT/test.log"
