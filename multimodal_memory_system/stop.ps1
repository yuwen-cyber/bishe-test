# ============================================================
#  Multimodal Memory System - one-click stop
#  Usage: double-click stop.bat, or run
#         powershell -NoProfile -ExecutionPolicy Bypass -File stop.ps1
# ============================================================

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidDir = Join-Path $root ".run"

Write-Host ""
Write-Host "  Multimodal Memory System - stopping..." -ForegroundColor Yellow
Write-Host ""

# [1] stop backend & frontend by saved PID (kills the whole console tree,
#     including the uvicorn --reload respawner / vite node process)
foreach ($name in 'backend', 'frontend') {
    $pf = Join-Path $pidDir "$name.pid"
    $pidVal = $null
    if (Test-Path $pf) {
        $pidVal = Get-Content $pf -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if ($pidVal -and (Get-Process -Id $pidVal -ErrorAction SilentlyContinue)) {
        taskkill /F /T /PID $pidVal 2>&1 | Out-Null
        Write-Host "  stopped $name (pid $pidVal)" -ForegroundColor Green
    } else {
        Write-Host "  $name : no saved process" -ForegroundColor DarkGray
    }
    Remove-Item $pf -ErrorAction SilentlyContinue
}

# [2] fallback: reliably kill backend/frontend servers. Loop a few times
#     because `uvicorn --reload` respawns its worker after it is killed, and
#     a backend started outside start.ps1 has no saved PID.
for ($i = 1; $i -le 8; $i++) {
    $ids = @()

    # backend reloader / server (python) - kill the root so its worker dies too
    $ids += Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'uvicorn|app\.main' } |
        Select-Object -ExpandProperty ProcessId

    # frontend vite dev server (node)
    $ids += Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'node.exe' -and $_.CommandLine -match 'vite' } |
        Select-Object -ExpandProperty ProcessId

    # anything still holding the ports (catches orphaned workers)
    foreach ($p in 5173, 8000) {
        $ids += Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess
    }

    $ids = @($ids | Where-Object { $_ } | Sort-Object -Unique)
    if ($ids.Count -eq 0) { break }

    foreach ($id in $ids) { taskkill /F /T /PID $id 2>$null | Out-Null }
    Write-Host "  killed server process(es): $($ids -join ', ')" -ForegroundColor Green
    Start-Sleep -Milliseconds 700
}

# final port status
foreach ($p in 5173, 8000) {
    $left = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($left) {
        Write-Host "  WARNING: port $p still in use" -ForegroundColor Yellow
    } else {
        Write-Host "  port $p released" -ForegroundColor Green
    }
}

# [3] stop docker middlewares (etcd / minio / milvus)
docker compose -f (Join-Path $root "docker-compose.yml") stop etcd minio milvus 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  docker middlewares stopped (etcd / minio / milvus)" -ForegroundColor Green
} else {
    Write-Host "  docker middlewares not reachable (Docker Desktop may already be off)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  Done. (local MySQL service is left running)" -ForegroundColor Green
Write-Host ""