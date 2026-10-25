#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /root/miniconda3/bin/python run_suite.py "$@"
