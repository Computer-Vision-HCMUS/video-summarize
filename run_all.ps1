# Run pipeline: download SumMe+TVSum -> prepare labels & videos -> extract features -> train -> eval (PowerShell)
# Uses current Python environment. If you see ModuleNotFoundError, exit .venv first: deactivate
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path

Write-Host "=== 1. Download SumMe+TVSum (Kaggle) and prepare labels + videos ===" -ForegroundColor Cyan
python -m scripts.download_datasets --all
# Manual: python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum --clear-dummy

Write-Host "=== 2. Extract CNN features from videos (skip if already from .h5) ===" -ForegroundColor Cyan
if (-not (Test-Path "data\features\_meta.json")) {
  python -m scripts.extract_features --config configs/config.yaml
} else { Write-Host "  (using existing features from .h5)" }

Write-Host "=== 3. Train ===" -ForegroundColor Cyan
python -m scripts.train --config configs/config.yaml

Write-Host "=== 4. Eval ===" -ForegroundColor Cyan
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt

Write-Host "=== Done ===" -ForegroundColor Green
