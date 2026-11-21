# ============================================================
#  Multimodal Memory System - one-click start
#  Usage: double-click start.bat, or run
#         powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
# ============================================================

$root        = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir  = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$nodeBin     = Join-Path $env:LOCALAPPDATA "Programs\nodejs\node-v22.12.0-win-x64"
$pidDir      = Join-Path $root ".run"

New-Item -ItemType Directory -Force -Path $pidDir | Out-Null

Write-Host ""
Write-Host "  Multimodal Memory System - starting..." -ForegroundColor Green
Write-Host ""

# [0] add portable Node to PATH (child processes inherit it)
if (Test-Path (Join-Path $nodeBin "node.exe")) {
    $env:Path = "$nodeBin;$env:Path"
} else {
    Write-Warning "[0/5] portable Node not found, frontend may fail: $nodeBin"
}

# [1] local MySQL check
if (Get-NetTCPConnection -LocalPort 3306 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "[1/5] MySQL is running (localhost:3306)" -ForegroundColor Green
} else {
    Write-Warning "[1/5] MySQL (3306) is NOT running. Start the MySQL service first (usually 'MySQL80')."
}

# [2] locate + start Docker Desktop, then wait until daemon is ready
$ddExe = $null
$reg = Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*','HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -like 'Docker Desktop' } | Select-Object -First 1
$candidates = @()
if ($reg -and $reg.InstallLocation) { $candidates += (Join-Path $reg.InstallLocation 'Docker Desktop.exe') }
$candidates += @(
    (Join-Path $env:LOCALAPPDATA 'Docker\Docker\Docker Desktop.exe'),
    (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Docker\Docker\Docker Desktop.exe')
)
foreach ($c in $candidates) { if (Test-Path $c) { $ddExe = $c; break } }

$dockerOk = $false
for ($i = 1; $i -le 60; $i++) {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerOk = $true; break }
    if ($i -eq 1 -and $ddExe) {
        Write-Host "[2/5] starting Docker Desktop ($ddExe) ..." -ForegroundColor Yellow
        Start-Process $ddExe -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}
if (-not $dockerOk) {
    if ($ddExe) { Write-Error "[2/5] Docker Desktop did not become ready in time. Start it manually, then re-run." }
    else { Write-Error "[2/5] Docker Desktop not found. Start it manually, then re-run." }
    exit 1
}
Write-Host "[2/5] Docker is ready" -ForegroundColor Green

# [3] start etcd / MinIO / Milvus
Write-Host "[3/5] starting etcd / MinIO / Milvus ..." -ForegroundColor Yellow
docker compose -f (Join-Path $root "docker-compose.yml") up -d etcd minio milvus
if ($LASTEXITCODE -ne 0) {
    Write-Error "[3/5] failed to start middlewares, see docker compose output above."
}

# [4] start backend (new window), save PID for stop.bat
Write-Host "[4/5] starting backend (http://localhost:8000)" -ForegroundColor Yellow
$backendCmd = ".\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"
$b = Start-Process powershell -PassThru -WorkingDirectory $backendDir -ArgumentList "-NoExit -ExecutionPolicy Bypass -Command ""$backendCmd"""
Set-Content -Path (Join-Path $pidDir "backend.pid") -Value $b.Id

# [5] start frontend (new window), save PID for stop.bat
Write-Host "[5/5] starting frontend (http://localhost:5173)" -ForegroundColor Yellow
$frontendCmd = "npm run dev"
$f = Start-Process powershell -PassThru -WorkingDirectory $frontendDir -ArgumentList "-NoExit -ExecutionPolicy Bypass -Command ""$frontendCmd"""
Set-Content -Path (Join-Path $pidDir "frontend.pid") -Value $f.Id

Write-Host ""
Write-Host "  Done. Frontend: http://localhost:5173   Backend: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  NOTE: first backend startup loads models (~20-40s), please wait a moment." -ForegroundColor Yellow
Write-Host ""

Start-Sleep -Seconds 8
Start-Process "http://localhost:5173"