$env:PYTHONPATH="."
Write-Host "Running Mamba-CfC with LAST pooling..."
.venv\Scripts\python.exe scripts/train.py --model mamba-cfc --pool_mode last --fast_dev_run

Write-Host "Running Mamba-CfC with MEAN pooling..."
.venv\Scripts\python.exe scripts/train.py --model mamba-cfc --pool_mode mean --fast_dev_run

Write-Host "Pooling confirmation test complete."
