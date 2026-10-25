#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run.sh
bash predict.sh --split dev
