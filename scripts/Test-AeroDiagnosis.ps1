[CmdletBinding()]
param(
    [string]$RuntimeDir = "",
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = Join-Path $ProjectRoot ".runtime"
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$env:UV_CACHE_DIR = Join-Path $RuntimeDir "uv-cache"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $RuntimeDir ".venv"
$RuntimeTempDir = Join-Path $RuntimeDir "tmp"
New-Item -ItemType Directory -Force $RuntimeTempDir | Out-Null
$env:AERODIAGNOSIS_RUNTIME_DIR = $RuntimeDir
$env:AERODIAGNOSIS_DATABASE_PATH = Join-Path (Join-Path $RuntimeDir "data") "aerodiagnosis.db"
$env:AERODIAGNOSIS_TEST_TMP = Join-Path $RuntimeDir "test-runs"
$Python = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime is not initialized. Run scripts\Initialize-AeroDiagnosis.ps1 first."
}

& $Python -c "from aerodiagnosis.config import RuntimeSettings; from aerodiagnosis.adapters.persistence import create_graph_store, create_vector_store; from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase; s=RuntimeSettings.from_env(); v=SQLiteDatabase(s.database_path).migrate(); print(f'schema={v} vector={create_vector_store(s).backend_name} graph={create_graph_store(s).backend_name}')"
if ($LASTEXITCODE -ne 0) {
    throw "Runtime self-check failed with exit code $LASTEXITCODE"
}

if ($Full) {
    Push-Location $ProjectRoot
    try {
        & uv run --frozen --no-sync ruff check src tests
        if ($LASTEXITCODE -ne 0) { throw "Ruff failed" }
        & uv run --frozen --no-sync mypy
        if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
        & $Python -m compileall -q code\python
        if ($LASTEXITCODE -ne 0) { throw "legacy compile check failed" }
        & uv run --frozen --no-sync pytest `
            -p no:cacheprovider `
            --cov=aerodiagnosis `
            --cov-report=term-missing
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
        $env:TEMP = $RuntimeTempDir
        $env:TMP = $RuntimeTempDir
        & $Python -m hatchling build
        if ($LASTEXITCODE -ne 0) { throw "package build failed" }
    }
    finally {
        Pop-Location
    }
}

Write-Host "AeroDiagnosis check passed."
