#!/usr/bin/env bash
# Chạy tuần tự: tải SumMe+TVSum -> labels & videos -> extract features -> train -> eval (Git Bash / WSL / Linux)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo "=== 1. Cài đặt (lần đầu) ==="
python -m venv .venv 2>/dev/null || true
if [ -f .venv/Scripts/activate ]; then
  . .venv/Scripts/activate
else
  . .venv/bin/activate
fi
pip install -e . -q

echo "=== 2. Tải dataset SumMe+TVSum (Kaggle) và chuẩn bị labels + videos ==="
python -m scripts.download_datasets --all
# Nếu tải tay: python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum --clear-dummy

echo "=== 3. Extract CNN features từ video ==="
python -m scripts.extract_features --config configs/config.yaml

echo "=== 4. Train ==="
python -m scripts.train --config configs/config.yaml

echo "=== 5. Eval ==="
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt

echo "=== Xong ==="
