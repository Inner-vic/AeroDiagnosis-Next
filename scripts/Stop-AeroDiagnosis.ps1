[CmdletBinding()]
param([string]$RuntimeDir = "")

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = Join-Path $ProjectRoot ".runtime"
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$PidFile = Join-Path $RuntimeDir "server.pid.json"

if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) {
    Write-Host "AeroDiagnosis is not running (no PID file)."
    exit 0
}

$Saved = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json
$Process = Get-Process -Id $Saved.pid -ErrorAction SilentlyContinue
if (-not $Process) {
    Remove-Item -LiteralPath $PidFile
    Write-Host "Removed stale PID file; no server process was running."
    exit 0
}

$ActualStart = $Process.StartTime.ToUniversalTime()
$ExpectedStart = ([DateTime]$Saved.started_at).ToUniversalTime()
if ([Math]::Abs(($ActualStart - $ExpectedStart).TotalSeconds) -gt 2) {
    throw "PID $($Saved.pid) belongs to a different process; refusing to stop it."
}

Stop-Process -Id $Process.Id
if (-not $Process.WaitForExit(10000)) {
    Stop-Process -Id $Process.Id -Force
}
Remove-Item -LiteralPath $PidFile
Write-Host "AeroDiagnosis stopped. Runtime data was preserved at $RuntimeDir."
