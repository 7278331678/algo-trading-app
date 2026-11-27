# Stops every pod but keeps all data (databases, MinIO, MLflow). start-local brings it back.
# Native tools (docker, kubectl, helm) are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
kubectl config use-context docker-desktop | Out-Null
kubectl -n trading scale deployment --all --replicas=0
kubectl -n trading scale statefulset --all --replicas=0
Write-Host "Stopped. Data is kept. Run start-local.bat to start again." -ForegroundColor Green
