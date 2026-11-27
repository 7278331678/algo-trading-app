# Deletes the whole local stack INCLUDING ALL DATA (downloaded candles, models, tokens).
# Native tools (docker, kubectl, helm) are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$answer = Read-Host "This deletes all local data in namespace 'trading'. Type RESET to continue"
if ($answer -ne "RESET") { Write-Host "Cancelled."; exit 0 }
kubectl config use-context docker-desktop | Out-Null
helm uninstall trading -n trading 2>$null
# Docker Desktop (kind mode) can leave LoadBalancer services stuck on a cleanup finalizer,
# which blocks namespace deletion forever. Clear it first.
$patch = Join-Path $env:TEMP "clear-finalizers.json"
Set-Content -Path $patch -Value '{"metadata":{"finalizers":null}}' -Encoding ascii
foreach ($svc in @(kubectl -n trading get svc -o name 2>$null)) {
    kubectl -n trading delete $svc --wait=false 2>$null | Out-Null
    kubectl -n trading patch $svc --type=merge --patch-file $patch 2>$null | Out-Null
}
Remove-Item $patch -ErrorAction SilentlyContinue
Write-Host "Deleting namespace (waits until volumes are released, up to 3 minutes)..."
kubectl delete namespace trading --wait=true --timeout=180s
if ($LASTEXITCODE -ne 0) { Write-Host "Namespace still terminating; start-local will finish the cleanup." -ForegroundColor Yellow }
Write-Host "Local stack removed. Run start-local.bat for a fresh start." -ForegroundColor Green
