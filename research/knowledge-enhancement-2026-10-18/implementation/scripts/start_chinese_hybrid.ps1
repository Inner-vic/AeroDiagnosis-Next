param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$taskRuntime = Join-Path $taskRoot '.runtime/chinese-hybrid-v1'
$taskProfile = Join-Path $taskRuntime 'profile'
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath (Join-Path $taskRuntime 'activation-receipt.json'))) {
    throw 'Run activate_chinese_hybrid_profile.py before starting this profile.'
}
$taskPort = & $taskPython -c 'from pathlib import Path; import sys; from aerodiagnosis.config import RuntimeSettings; print(RuntimeSettings.from_env(Path(sys.argv[1])).api_port)' $taskProfile
if ($LASTEXITCODE -ne 0) { throw 'Profile validation failed.' }
$taskPortOpen = & $taskPython -c 'import socket,sys; s=socket.socket(); s.settimeout(1); print(s.connect_ex(("127.0.0.1",int(sys.argv[1]))) == 0); s.close()' $taskPort
if ($taskPortOpen -eq 'True') { throw "Port $taskPort already in use; existing process retained." }
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-X','utf8','scripts/run_upstream_application.py','--profile-root',"`"$taskProfile`"") -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRuntime 'server.stdout.log') -RedirectStandardError (Join-Path $taskRuntime 'server.stderr.log') -PassThru
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRuntime 'server.pid')
Write-Output "Chinese hybrid application starting on loopback port $taskPort; PID $($taskProcess.Id)."
