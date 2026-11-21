# 分批驱动：分批调用 rebuild_multimodal_real.py，每批结束退出进程释放内存。
# 首批（start=0）清空 Milvus 集合；后续批次 --keep-collection 续灌。
# 用法（backend 目录）：
#   .\.venv\Scripts\python.exe 由本脚本内部调用；直接运行：
#   powershell -ExecutionPolicy Bypass -File experiments\rebuild_batches.ps1
#   可选参数：-BatchSize 40

param(
    [int]$BatchSize = 50,
    [int]$Total = 700
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
$outDir = "experiments\output"
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$allFailed = @()
$start = 0
$batchNo = 0

while ($start -lt $Total) {
    $lim = [Math]::Min($BatchSize, $Total - $start)
    $pyArgs = @("experiments\rebuild_multimodal_real.py", "--start", $start, "--limit", $lim)
    if ($start -gt 0) { $pyArgs += "--keep-collection" }

    $batchNo++
    Write-Host "=== 批次 $batchNo : start=$start limit=$lim ==="
    & $py @pyArgs 2>&1 | Tee-Object -FilePath (Join-Path $logDir "rebuild_batch_$start.log")

    $failedFile = Join-Path $outDir "rebuild_real_failed.json"
    if (Test-Path $failedFile) {
        $batchFailed = Get-Content $failedFile -Raw -Encoding UTF8
        if ($batchFailed.Trim() -ne "[]") {
            $allFailed += $batchFailed
            Copy-Item $failedFile (Join-Path $outDir "rebuild_failed_batch_$start.json") -Force
        }
    }
    $start += $lim
}

$merged = @()
foreach ($f in $allFailed) { $merged += ($f | ConvertFrom-Json) }
$merged | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outDir "rebuild_real_failed.json") -Encoding UTF8
Write-Host "=== 全部分批结束，累计失败 $($merged.Count) 条 ==="
