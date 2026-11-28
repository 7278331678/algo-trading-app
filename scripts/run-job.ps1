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
# no extra arguments: keep the template's own (e.g. signal-check's --decision-times 09:45);
# extra arguments: they replace everything after the command name
if ($extra.Count -gt 0) { $spec.template.spec.containers[0].args = @($base[0]) + $extra }
$shown = @($spec.template.spec.containers[0].args)
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

Write-Host "`nJob $name started: python -m app.jobs $($shown -join ' ')" -ForegroundColor Cyan
Write-Host "Waiting for its container to start..."
$phase = ""
for ($i = 0; $i -lt 150; $i++) {   # up to 5 minutes
    $phase = kubectl -n $Namespace get pods -l job-name=$name -o jsonpath="{.items[0].status.phase}" 2>$null
    if ($phase -and $phase -ne "Pending") { break }
    Start-Sleep -Seconds 2
}
if (-not $phase -or $phase -eq "Pending") {
    Write-Host "Still not started after 5 minutes. Check with: kubectl -n $Namespace describe pod -l job-name=$name" -ForegroundColor Yellow
    exit 1
}
Write-Host "Showing its log (Ctrl+C stops watching, not the job)...`n"
kubectl -n $Namespace logs -f job/$name
Write-Host "`nTo see the end again later: kubectl -n $Namespace logs job/$name --tail=3"
