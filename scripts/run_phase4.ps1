$env:PYTHONPATH="."
$models = @("mamba-only", "cfc-only", "mamba-cfc", "mamba-3l", "mamba-width")
$pool_mode = "last"
$epochs = 50

foreach ($model in $models) {
    Write-Host "========================================="
    Write-Host "Training $model"
    Write-Host "========================================="
    
    .venv\Scripts\python.exe scripts\train.py --model $model --pool_mode $pool_mode --epochs $epochs --patience 5 --batch_size 256
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed for $model. Exiting."
        exit $LASTEXITCODE
    }
}

Write-Host "========================================="
Write-Host "Evaluating all models"
Write-Host "========================================="

.venv\Scripts\python.exe scripts\evaluate.py --pool_mode $pool_mode

Write-Host "Phase 4 Complete!"
