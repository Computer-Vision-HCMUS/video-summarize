#!/usr/bin/env bash
# Run pipeline: download SumMe+TVSum -> labels & videos -> extract features -> train -> eval (Git Bash / WSL / Linux)
# Uses current Python environment. To exit venv: deactivate
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo "=== 1. Download SumMe+TVSum (Kaggle) and prepare labels + videos ==="
python -m scripts.download_datasets --all
# Manual: python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum --clear-dummy

echo "=== 2. Extract CNN features from videos (skip if already from .h5) ==="
[ -f data/features/_meta.json ] && echo "  (using existing features from .h5)" || python -m scripts.extract_features --config configs/config.yaml

echo "=== 3. Train ==="
python -m scripts.train --config configs/config.yaml

echo "=== 4. Eval ==="
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt

echo "=== Done ==="
