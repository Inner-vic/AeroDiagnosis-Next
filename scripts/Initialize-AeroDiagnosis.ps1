[CmdletBinding()]
param(
    [string]$RuntimeDir = "",
    [switch]$WithoutDevTools
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = Join-Path $ProjectRoot ".runtime"
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$UvCacheDir = Join-Path $RuntimeDir "uv-cache"
$VenvDir = Join-Path $RuntimeDir ".venv"
$DataDir = Join-Path $RuntimeDir "data"
$LogDir = Join-Path $RuntimeDir "logs"
$RuntimeTempDir = Join-Path $RuntimeDir "tmp"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv from https://docs.astral.sh/uv/ and rerun this script."
}

New-Item -ItemType Directory -Force `
    $RuntimeDir, $UvCacheDir, $DataDir, $LogDir, $RuntimeTempDir | Out-Null
$env:UV_CACHE_DIR = $UvCacheDir
$env:UV_PROJECT_ENVIRONMENT = $VenvDir
$env:TEMP = $RuntimeTempDir
$env:TMP = $RuntimeTempDir
$env:AERODIAGNOSIS_RUNTIME_DIR = $RuntimeDir
$env:AERODIAGNOSIS_DATABASE_PATH = Join-Path $DataDir "aerodiagnosis.db"

$SyncArguments = @("sync", "--frozen", "--python", "3.13")
if (-not $WithoutDevTools) {
    $SyncArguments += @("--group", "dev")
}
& uv @SyncArguments
if ($LASTEXITCODE -ne 0) {
    throw "uv sync failed with exit code $LASTEXITCODE"
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -c "from aerodiagnosis.config import RuntimeSettings; from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase; s=RuntimeSettings.from_env(); s.prepare(); print('schema', SQLiteDatabase(s.database_path).migrate())"
if ($LASTEXITCODE -ne 0) {
    throw "SQLite initialization failed with exit code $LASTEXITCODE"
}

Write-Host "AeroDiagnosis initialized."
Write-Host "  Runtime: $RuntimeDir"
Write-Host "  Virtual environment: $VenvDir"
Write-Host "  Database: $env:AERODIAGNOSIS_DATABASE_PATH"
