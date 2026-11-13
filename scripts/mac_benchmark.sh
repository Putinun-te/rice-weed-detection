#!/usr/bin/env bash
# Phase 6: CPU benchmark and a short CSV inference on another computer (written for the MacBook).
#
# 1. On the PC, the model bundle is bundles/phase6_models.tar.gz (not in git). Copy it to this computer.
# 2. Here:  git pull && bash scripts/mac_benchmark.sh ~/Downloads/phase6_models.tar.gz
# 3. Send back the two files it names at the end.
#
# The numbers describe this computer's CPU only. They are not a Raspberry Pi or on-drone result.
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE=${1:?usage: bash scripts/mac_benchmark.sh <path to phase6_models.tar.gz>}

if [ ! -x .venv/bin/python ]; then
  bash scripts/setup_env.sh
fi
.venv/bin/python -m pip install -q -r requirements.txt -r requirements-export.txt

rm -rf phase6 && mkdir -p phase6
tar -xzf "$BUNDLE" -C phase6
NAME="$(uname -s)_$(uname -m)"
.venv/bin/python -m rwd.benchmark --weights phase6/best.pt phase6/best.onnx phase6/best_ncnn_model \
  --images phase6/images --n 20 --out "phase6/benchmark_cpu_${NAME}.json"
.venv/bin/python -m rwd.infer --weights phase6/best.onnx --source phase6/images --out "phase6/detections_${NAME}.csv"
echo
echo "Done. Send these two files back:"
echo "  $(pwd)/phase6/benchmark_cpu_${NAME}.json"
echo "  $(pwd)/phase6/detections_${NAME}.csv"
