[CmdletBinding()]
param(
    [string]$RuntimeDir = "",
    [int]$Port = 8080,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $RuntimeDir = Join-Path $ProjectRoot ".runtime"
}
$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$VenvDir = Join-Path $RuntimeDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$LogDir = Join-Path $RuntimeDir "logs"
$PidFile = Join-Path $RuntimeDir "server.pid.json"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime is not initialized. Run scripts\Initialize-AeroDiagnosis.ps1 first."
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}

New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:UV_CACHE_DIR = Join-Path $RuntimeDir "uv-cache"
$env:UV_PROJECT_ENVIRONMENT = $VenvDir
$RuntimeTempDir = Join-Path $RuntimeDir "tmp"
New-Item -ItemType Directory -Force $RuntimeTempDir | Out-Null
$env:AERODIAGNOSIS_RUNTIME_DIR = $RuntimeDir
$env:AERODIAGNOSIS_DATABASE_PATH = Join-Path (Join-Path $RuntimeDir "data") "aerodiagnosis.db"
$env:AERODIAGNOSIS_API_HOST = "127.0.0.1"
$env:AERODIAGNOSIS_API_PORT = $Port.ToString()
$ServerArguments = @("-m", "uvicorn", "aerodiagnosis.adapters.api:app", "--host", "127.0.0.1", "--port", $Port.ToString())

if ($Foreground) {
    Push-Location $ProjectRoot
    try { & $Python @ServerArguments }
    finally { Pop-Location }
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $PidFile) {
    $Existing = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json
    $ExistingProcess = Get-Process -Id $Existing.pid -ErrorAction SilentlyContinue
    if ($ExistingProcess) {
        Write-Host "AeroDiagnosis is already running with PID $($Existing.pid)."
        exit 0
    }
}

$Process = Start-Process -FilePath $Python `
    -ArgumentList $ServerArguments `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "server.stdout.log") `
    -RedirectStandardError (Join-Path $LogDir "server.stderr.log") `
    -PassThru
@{
    pid = $Process.Id
    started_at = $Process.StartTime.ToUniversalTime().ToString("o")
    port = $Port
} | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding utf8

$Ready = $false
for ($Attempt = 0; $Attempt -lt 15; $Attempt++) {
    Start-Sleep -Milliseconds 500
    if ($Process.HasExited) { break }
    try {
        $Response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        if ($Response.status -eq "ok") { $Ready = $true; break }
    }
    catch { }
}
if (-not $Ready) {
    throw "Server did not become ready. Inspect $LogDir."
}
Write-Host "AeroDiagnosis started: http://127.0.0.1:$Port (PID $($Process.Id))"
