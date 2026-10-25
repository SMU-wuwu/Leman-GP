#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1
export LEMAN_GP_WORLD_SIZE=2
export OMP_NUM_THREADS=2
for seed in 42 43; do
  for variant in F attention_norm_only uniform_attention; do
    target="/root/data/WorkSpace/output/leman_attention_mechanism_v2/${variant}_seed${seed}"
    if [[ -f "$target/COMPLETE.json" ]]; then
      continue
    fi
    /root/miniconda3/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --root /root/data/WorkSpace --variant "$variant" --seed "$seed"
  done
done
/root/miniconda3/bin/python summarize_extension.py --root /root/data/WorkSpace
