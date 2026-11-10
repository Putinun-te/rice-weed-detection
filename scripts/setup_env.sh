#!/usr/bin/env bash
# Create the project virtual env (.venv) and a Jupyter kernel "Python (rwd)".
#
# Why a separate env: installing into Studio Lab's default env changed pytest and broke
# amzn-sagemaker-scheduler's pinned version. The project env keeps the default env untouched.
#
# Safe to run again. Usage:
#   bash ~/rice-weed-detection/scripts/setup_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY_BASE=${PY_BASE:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "== create .venv with $($PY_BASE --version 2>&1)"
  "$PY_BASE" -m venv .venv
fi

echo "== install requirements into .venv"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt ipykernel

echo "== register Jupyter kernel 'Python (rwd)'"
.venv/bin/python -m ipykernel install --user --name rwd --display-name "Python (rwd)" >/dev/null

echo "== env ready: $(.venv/bin/python --version 2>&1) at $(pwd)/.venv"
