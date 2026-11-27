# Starts the historical data download as a Kubernetes Job and follows its log.
# Usage: backfill.bat                       (everything: all stocks, all intervals)
#        backfill.bat --symbols INFY TCS     (only some stocks)
#        backfill.bat --intervals day        (only daily candles)
#        backfill.bat --refresh --symbols X  (delete and re-download, e.g. after a split)
# The job keeps running if you close this window; run it again to resume after an interruption.
$ErrorActionPreference = "Continue"
$Namespace = "trading"
$ApiPort = 8100
$extra = $args

function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

kubectl config use-context docker-desktop | Out-Null

# 1. Zerodha session must exist (the download uses today's access token)
try {
    $st = Invoke-RestMethod -Uri "http://localhost:$ApiPort/auth/kite/status" -TimeoutSec 5
} catch { Fail "API not reachable on http://localhost:$ApiPort. Run start-local.bat first." }
if (-not $st.logged_in) {
    Start-Process "http://localhost:$ApiPort/auth/kite/login"
    Fail "Not logged in to Zerodha. A login page was opened; log in, then run backfill.bat again."
}
Write-Host "Zerodha session OK (user $($st.user_id), valid until $($st.expires_at))"

# 2a. If a backfill is already running (e.g. the window was closed), just follow it
$running = @(kubectl -n $Namespace get jobs -o json | ConvertFrom-Json).items |
    Where-Object { $_.metadata.name -like "backfill-*" -and $_.status.active -gt 0 } |
    Sort-Object { $_.metadata.creationTimestamp } | Select-Object -Last 1
if ($running) {
    $rn = $running.metadata.name
    Write-Host "A backfill is already running ($rn). Following its progress instead of starting another.`n" -ForegroundColor Cyan
    kubectl -n $Namespace logs -f job/$rn
    exit 0
}

# 2. Build a Job from the suspended 'backfill' CronJob, with any extra arguments
$cj = kubectl -n $Namespace get cronjob backfill -o json | ConvertFrom-Json
if (-not $cj) { Fail "CronJob 'backfill' not found. Run start-local.bat to deploy the latest version." }
$name = "backfill-" + (Get-Date -Format "yyyyMMdd-HHmmss")
$spec = $cj.spec.jobTemplate.spec
$spec.template.spec.containers[0].args = @("backfill") + $extra
$job = [ordered]@{
    apiVersion = "batch/v1"; kind = "Job"
    metadata = @{ name = $name; namespace = $Namespace; labels = @{ "app.kubernetes.io/part-of" = "trading-app" } }
    spec = $spec
}
$tmp = Join-Path $env:TEMP "$name.json"
# write UTF-8 without BOM (Windows PowerShell 5.1 adds a BOM that kubectl rejects)
[IO.File]::WriteAllText($tmp, ($job | ConvertTo-Json -Depth 50))
kubectl apply -f $tmp
$rc = $LASTEXITCODE
Remove-Item $tmp -ErrorAction SilentlyContinue
if ($rc -ne 0) { Fail "could not create the job" }

# 3. Follow the log until the job ends
Write-Host "`nJob $name started. Showing progress (Ctrl+C stops watching, not the job)...`n" -ForegroundColor Cyan
kubectl -n $Namespace wait --for=condition=ready pod -l job-name=$name --timeout=180s | Out-Null
kubectl -n $Namespace logs -f job/$name
kubectl -n $Namespace wait --for=condition=complete job/$name --timeout=10s 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBackfill finished. Coverage and quality: http://localhost:$ApiPort/data/status" -ForegroundColor Green
} else {
    Write-Host "`nJob did not report success. Check: kubectl -n $Namespace get pods -l job-name=$name" -ForegroundColor Yellow
    Write-Host "Re-running backfill.bat resumes where it stopped."
}
