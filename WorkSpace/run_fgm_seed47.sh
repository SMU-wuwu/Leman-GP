#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/data/WorkSpace/leman_gp_fgm_seed47_dual4090
exec bash run.sh "$@"
