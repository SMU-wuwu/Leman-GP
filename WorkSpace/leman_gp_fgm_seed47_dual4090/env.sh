#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
export LEMAN_GP_ROOT=/root/data/WorkSpace
export LEMAN_GP_SEED=47
export LEMAN_GP_OUTPUT=/root/data/WorkSpace/output/leman_fgm_seed47_dual4090
export LEMAN_GP_WARM_START=/root/data/WorkSpace/output/leman_finetune_v4_seed42/best.pt
export LEMAN_GP_WORLD_SIZE=2
export PYTHONHASHSEED="$LEMAN_GP_SEED"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LEMAN_GP_OUTPUT"
