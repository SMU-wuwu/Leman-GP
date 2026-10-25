#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
LEBERT_PYTHON="${LEBERT_PYTHON:-/root/miniconda3/bin/python}"
LEBERT_ROOT="${LEBERT_ROOT:-/root/data/WorkSpace}"
if (( $# == 0 )); then set -- 42 43 44; fi
for seed in "$@"; do
    mkdir -p "$LEBERT_ROOT/output/lebert_agcner_seed$seed"
    "$LEBERT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
      --root "$LEBERT_ROOT" --seed "$seed" --epochs 20 --batch-size 16 --eval-batch-size 32 \
      --encoder-lr 1e-5 --head-lr 1e-4 --warmup 0.1 --save-steps 250 \
      2>&1 | tee -a "$LEBERT_ROOT/output/lebert_agcner_seed$seed/train.log"
done
