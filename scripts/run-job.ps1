# Runs one of the batch jobs (CronJob templates) once, with optional extra arguments, and follows its log.
# Usage: run-job.bat <job> [args...]
#   run-job.bat train-models --decision-times 09:45 --barrier daily_0.6x1.2
#   run-job.bat backtest --strategies rules --slippage-bps 2
#   run-job.bat build-features
# The job keeps running if you close this window (follow it again with: kubectl -n trading logs -f job/<name>).
$ErrorActionPreference = "Continue"
$Namespace = "trading"
if ($args.Count -lt 1) { Write-Host "Usage: run-job.bat <job> [args...]"; exit 1 }
$cron = $args[0]
$extra = @()
if ($args.Count -gt 1) { $extra = $args[1..($args.Count - 1)] }

kubectl config use-context docker-desktop | Out-Null
$cj = kubectl -n $Namespace get cronjob $cron -o json | ConvertFrom-Json
if (-not $cj) { Write-Host "No job template named '$cron'. Run start-local.bat to deploy the latest version." -ForegroundColor Red; exit 1 }

$name = ($cron + "-" + (Get-Date -Format "MMdd-HHmmss")).ToLower()
$spec = $cj.spec.jobTemplate.spec
$base = @($spec.template.spec.containers[0].args)
$spec.template.spec.containers[0].args = @($base[0]) + $extra
$job = [ordered]@{
    apiVersion = "batch/v1"; kind = "Job"
    metadata = @{ name = $name; namespace = $Namespace; labels = @{ "app.kubernetes.io/part-of" = "trading-app" } }
    spec = $spec
}
$tmp = Join-Path $env:TEMP "$name.json"
[IO.File]::WriteAllText($tmp, ($job | ConvertTo-Json -Depth 50))   # UTF-8 without BOM
kubectl apply -f $tmp
$rc = $LASTEXITCODE
Remove-Item $tmp -ErrorAction SilentlyContinue
if ($rc -ne 0) { Write-Host "could not create the job" -ForegroundColor Red; exit 1 }

Write-Host "`nJob $name started: python -m app.jobs $($base[0]) $($extra -join ' ')" -ForegroundColor Cyan
Write-Host "Showing its log (Ctrl+C stops watching, not the job)...`n"
kubectl -n $Namespace wait --for=condition=ready pod -l job-name=$name --timeout=300s | Out-Null
kubectl -n $Namespace logs -f job/$name
