#!/usr/bin/env bash
# Phase 2 on SageMaker Studio Lab: install, test, download WeedyRice, inspect raw files.
# A CPU runtime is enough. Keep the GPU quota for training.
#
# Run from anywhere, in the background so a closed browser tab does not stop it:
#   nohup bash ~/rice-weed-detection/scripts/studiolab_data.sh > ~/rice-weed-detection/studiolab_data.log 2>&1 &
#   tail -f ~/rice-weed-detection/studiolab_data.log
#
# Safe to run again: it skips the download if the RGB folder already exists.
# At the end, send the content of data/inspect.txt back for the config.

set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw/weedyrice
NEED_GB=7

echo "== $(date '+%Y-%m-%d %H:%M:%S') start in $(pwd)"

# Project env in .venv, so Studio Lab's default env is not changed.
bash scripts/setup_env.sh
PY=.venv/bin/python

echo "== unit tests (synthetic data only)"
$PY -m pytest -q

echo "== disk"
df -h "$HOME" | tail -1
free_gb=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')

if [ -d "$RAW/RGB" ]; then
  echo "== $RAW/RGB already exists, skipping download"
else
  if [ "$free_gb" -lt "$NEED_GB" ]; then
    echo "ERROR: need at least ${NEED_GB} GB free, have ${free_gb} GB. Free space and run again."
    exit 1
  fi
  echo "== download WeedyRice-RGBMS-DB (about 5.3 GB zip, CC BY 4.0, see DATASET.md)"
  $PY -m rwd.download mendeley --dataset-id vt4s83pxx6 --version 1 \
    --out "$RAW" --include RGB Masks Metadata
fi

echo "== inspect raw files (also saved to data/inspect.txt)"
{
  $PY -m rwd.prepare_weedyrice --config configs/weedyrice.yaml --inspect
  echo
  for d in RGB Masks Metadata; do
    if [ -d "$RAW/$d" ]; then
      echo "$d: $(find "$RAW/$d" -type f | wc -l) files, first 3:"
      find "$RAW/$d" -type f | sort | head -3 | sed 's/^/  /'
    else
      echo "$d: missing"
    fi
  done
  echo
  echo "disk after download:"
  du -sh "$RAW" 2>/dev/null || true
  df -h "$HOME" | tail -1
} | tee data/inspect.txt

echo "== $(date '+%Y-%m-%d %H:%M:%S') done. Send data/inspect.txt back."
