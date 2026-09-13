[CmdletBinding()]
param(
    [string]$RuntimeRoot = "",
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Join-Path $ProjectRoot ".runtime\datasets\public-aero-corpus-v1"
}
$Python = Join-Path $ProjectRoot ".runtime\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime is not initialized. Run scripts\Initialize-AeroDiagnosis.ps1 first."
}
$Arguments = @((Join-Path $PSScriptRoot "build_public_corpus.py"), "--runtime-root", $RuntimeRoot)
if ($Offline) {
    $Arguments += "--offline"
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Public corpus build failed with exit code $LASTEXITCODE"
}

Write-Host "Public corpus is ready: $RuntimeRoot"
