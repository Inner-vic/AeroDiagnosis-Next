[CmdletBinding()]
param(
    [string]$Benchmark = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Benchmark)) {
    $Benchmark = Join-Path $ProjectRoot "evaluation\retrieval\benchmark-v1.json"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot ".runtime\experiments\retrieval"
}
$RuntimeDir = Join-Path $ProjectRoot ".runtime\retrieval-experiment"
$Python = Join-Path $ProjectRoot ".runtime\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime is not initialized. Run scripts\Initialize-AeroDiagnosis.ps1 first."
}

& $Python -m aerodiagnosis.retrieval_experiment $Benchmark $OutputDir --runtime-dir $RuntimeDir
if ($LASTEXITCODE -ne 0) {
    throw "Retrieval experiment failed with exit code $LASTEXITCODE"
}

Write-Host "Retrieval experiment results: $OutputDir"
