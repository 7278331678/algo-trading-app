# One-click local start for Docker Desktop Kubernetes.
# Run via start-local.bat (double-click) or: powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
# Native tools (docker, kubectl, helm) are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Namespace = "trading"
$Release = "trading"
$ApiPort = 8100

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

function Clear-StuckFinalizers([string]$ns) {
    # Docker Desktop (kind mode) can leave LoadBalancer services on a cleanup finalizer forever.
    $patch = Join-Path $env:TEMP "clear-finalizers.json"
    Set-Content -Path $patch -Value '{"metadata":{"finalizers":null}}' -Encoding ascii
    foreach ($obj in @(kubectl -n $ns get pvc,svc -o name 2>$null)) {
        $del = kubectl -n $ns get $obj -o jsonpath="{.metadata.deletionTimestamp}" 2>$null
        if ($del) { kubectl -n $ns patch $obj --type=merge --patch-file $patch 2>$null | Out-Null }
    }
    Remove-Item $patch -ErrorAction SilentlyContinue
}

function Wait-NamespaceGone([string]$ns) {
    # A deleted namespace stays "Terminating" until its volumes are released. Wait for it,
    # and after ~30s force-clear stuck pods and volume finalizers.
    for ($i = 0; $i -lt 100; $i++) {
        $phase = kubectl get namespace $ns -o jsonpath="{.status.phase}" 2>$null
        if (-not $phase -or $phase -ne "Terminating") { return }
        if ($i -eq 0) { Write-Host "Namespace '$ns' is still being deleted; waiting..." }
        if ($i -eq 10) {
            Write-Host "Still terminating after ~30s: clearing stuck pods, volume claims and services"
            kubectl -n $ns delete pods --all --grace-period=0 --force 2>$null | Out-Null
            $patch = Join-Path $env:TEMP "clear-finalizers.json"
            Set-Content -Path $patch -Value '{"metadata":{"finalizers":null}}' -Encoding ascii
            # LoadBalancer services in Docker Desktop kind mode can keep a cleanup finalizer forever
            foreach ($obj in @(kubectl -n $ns get pvc,svc -o name 2>$null)) {
                kubectl -n $ns patch $obj --type=merge --patch-file $patch 2>$null | Out-Null
            }
            Remove-Item $patch -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 3
    }
    Fail "Namespace '$ns' is stuck terminating. Run: kubectl get namespace $ns -o yaml  and send me the output."
}

# ---------------------------------------------------------------- 1. checks
Step "Checking tools"
foreach ($tool in @("docker", "kubectl", "helm")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $hint = @{ docker = "Install Docker Desktop"; kubectl = "Enable Kubernetes in Docker Desktop settings";
                   helm = "Run: winget install Helm.Helm  (then open a new terminal)" }[$tool]
        Fail "$tool not found. $hint"
    }
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop is not running. Start it and try again." }

$ctx = kubectl config current-context
if ($ctx -ne "docker-desktop") {
    Write-Host "Switching kubectl context from '$ctx' to 'docker-desktop'"
    kubectl config use-context docker-desktop | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "Kubernetes is not enabled in Docker Desktop (Settings > Kubernetes)." }
}
kubectl get nodes *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop Kubernetes is not ready yet. Wait for the green 'Kubernetes running' status." }

# ---------------------------------------------------------------- 2. .env
Step "Checking .env"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Warn ".env was missing, so it was created from .env.example."
    Write-Host "Open .env, fill in KITE_API_KEY, KITE_API_SECRET and the passwords, then run this again."
    notepad ".env"
    exit 1
}
$envMap = @{}
Get-Content ".env" | Where-Object { $_ -match '^\s*[A-Z0-9_]+\s*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; $envMap[$k.Trim()] = $v.Trim()
}
foreach ($key in @("POSTGRES_PASSWORD", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD")) {
    if (-not $envMap[$key] -or $envMap[$key] -like "change-me*") { Fail "Set $key in .env." }
}
if (-not $envMap["KITE_API_KEY"]) { Warn "KITE_API_KEY is empty: the stack will run, but Zerodha login is disabled." }

# ---------------------------------------------------------------- 3. Ollama
Step "Checking Ollama (local LLM)"
$model = if ($envMap["OLLAMA_MODEL"]) { $envMap["OLLAMA_MODEL"] } else { "llama3.1:8b" }
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
    if (-not ($tags.models | Where-Object { $_.name -eq $model })) {
        Write-Host "Pulling $model (first time only, a few GB)..."
        ollama pull $model
    } else { Write-Host "Ollama OK, model $model present" }
} catch {
    Warn "Ollama is not reachable on localhost:11434. LLM features will be off until you start it."
}

# ---------------------------------------------------------------- 4. build
Step "Building backend image"
$tag = "local-" + (Get-Date -Format "yyyyMMddHHmmss")
# no provenance/SBOM attestations: they break importing the image into kind nodes
docker build --provenance=false --sbom=false -t "trading-backend:$tag" backend
if ($LASTEXITCODE -ne 0) { Fail "docker build failed" }

# ---------------------------------------------------------------- 4b. image -> cluster
Step "Making the backend image visible to Kubernetes"
$nodes = @((kubectl get nodes -o jsonpath="{.items[*].metadata.name}") -split ' ' | Where-Object { $_ })
if ($nodes -contains "docker-desktop") {
    Write-Host "kubeadm cluster: Kubernetes shares Docker's image store."
} else {
    Write-Host "kind cluster (nodes: $($nodes -join ', ')): loading the image into each node"
    $tar = Join-Path $env:TEMP "trading-backend-$tag.tar"
    docker save -o $tar "trading-backend:$tag"
    if ($LASTEXITCODE -ne 0) { Fail "docker save failed" }
    foreach ($n in $nodes) {
        # cmd's "<" streams the tar byte-for-byte (PowerShell pipes would corrupt binary data)
        cmd /c "docker exec -i $n ctr -n k8s.io images import --digests - < `"$tar`""
        if ($LASTEXITCODE -ne 0) {
            Remove-Item $tar -ErrorAction SilentlyContinue
            Fail ("Could not load the image into node '$n'. In Docker Desktop: Settings > Kubernetes > " +
                  "Cluster provisioning method > choose 'kubeadm', Apply & restart, then run this again.")
        }
        $seen = docker exec $n crictl images 2>$null | Select-String "trading-backend\s+$tag"
        if (-not $seen) { Fail "image was imported but node '$n' does not list trading-backend:$tag" }
        Write-Host "  loaded into $n"
    }
    Remove-Item $tar -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- 5. deploy
Step "Applying namespace and secrets"
Wait-NamespaceGone $Namespace
kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f - | Out-Null
kubectl -n $Namespace create secret generic trading-app-secrets --from-env-file=.env --dry-run=client -o yaml |
    kubectl apply -f - | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "could not create the secret from .env (no quotes around values)" }

Step "Deploying with Helm (first run pulls images, can take several minutes)"
# A first install that failed leaves a release with nothing deployed; remove it so we can retry cleanly.
$history = helm history $Release -n $Namespace -o json 2>$null | ConvertFrom-Json
if ($history -and -not ($history | Where-Object { $_.status -in @("deployed", "superseded") })) {
    Write-Host "Removing the earlier failed install"
    helm uninstall $Release -n $Namespace --timeout 2m 2>$null | Out-Null
    Start-Sleep -Seconds 5
    Clear-StuckFinalizers $Namespace
    # make sure volumes, services and deployments are really gone before reinstalling
    kubectl -n $Namespace wait --for=delete deploy,svc,pvc -l app.kubernetes.io/part-of=trading-app --timeout=180s 2>$null | Out-Null
}
helm upgrade --install $Release deploy/helm/trading-app -n $Namespace `
    -f deploy/helm/trading-app/values-local.yaml `
    --set backend.image.tag=$tag --wait --timeout 15m
if ($LASTEXITCODE -ne 0) {
    kubectl -n $Namespace get pods
    if (kubectl -n $Namespace get pods | Select-String "ErrImageNeverPull") {
        Warn ("Kubernetes cannot see the locally built image. In Docker Desktop: Settings > General > " +
              "untick 'Use containerd for pulling and storing images', Apply & restart, then run this again.")
    }
    Write-Host "`nLast backend log lines (readiness failures show the exact error):" -ForegroundColor Yellow
    kubectl -n $Namespace logs deploy/backend --tail=15 2>$null
    Fail "Helm deploy did not become healthy. See the pod list above; 'kubectl -n trading logs deploy/backend' shows backend logs."
}

# ---------------------------------------------------------------- 6. verify
Step "Waiting for the API on localhost (in kind mode the load balancer can take 1-2 minutes)"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:$ApiPort/health/ready" -TimeoutSec 3
        if ($r.status -eq "ok") { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 3
}
kubectl -n $Namespace get pods
if (-not $ok) {
    kubectl -n $Namespace get svc
    Warn ("Pods are up but http://localhost:$ApiPort is not reachable yet. If EXTERNAL-IP above shows <pending>, " +
          "run this in a separate window and keep it open:  kubectl -n trading port-forward svc/backend ${ApiPort}:${ApiPort}")
    exit 1
}

Write-Host "`nAll services are up." -ForegroundColor Green
Write-Host "  API docs      http://localhost:$ApiPort/docs"
Write-Host "  System status http://localhost:$ApiPort/health/system"
Write-Host "  Zerodha login http://localhost:$ApiPort/auth/kite/login"
Write-Host "  MLflow        http://localhost:5000"
Write-Host "  MinIO console http://localhost:9001"
Write-Host "  Mail inbox    http://localhost:8025"
Start-Process "http://localhost:$ApiPort/docs"
