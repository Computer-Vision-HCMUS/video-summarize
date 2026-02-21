# Chạy tuần tự: tải SumMe+TVSum -> chuẩn bị labels & videos -> extract features -> train -> eval (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path

Write-Host "=== 1. Cài đặt (lần đầu) ===" -ForegroundColor Cyan
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -e . -q

Write-Host "=== 2. Tải dataset SumMe+TVSum (Kaggle) và chuẩn bị labels + videos ===" -ForegroundColor Cyan
python -m scripts.download_datasets --all
# Nếu không dùng Kaggle, tải tay từ Kaggle rồi chạy:
#   python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum --clear-dummy

Write-Host "=== 3. Extract CNN features từ video ===" -ForegroundColor Cyan
python -m scripts.extract_features --config configs/config.yaml

Write-Host "=== 4. Train ===" -ForegroundColor Cyan
python -m scripts.train --config configs/config.yaml

Write-Host "=== 5. Eval ===" -ForegroundColor Cyan
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt

Write-Host "=== Xong ===" -ForegroundColor Green
