param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$runtime = Join-Path $resolvedRoot '.runtime/neo4j'
$java = Join-Path $runtime 'runtime/zulu21.52.203-ca-jdk21.0.12.1-win_x64/bin/java.exe'
if (-not (Test-Path -LiteralPath $java)) { throw 'Portable JDK not prepared.' }
if (-not (Test-Path -LiteralPath (Join-Path $runtime 'build/classes/LocalNeo4j.class'))) { throw 'Build scripts/neo4j-runtime/pom.xml with Maven first.' }
$probe = [System.Net.Sockets.TcpClient]::new()
try { $probe.Connect('127.0.0.1', 7687); throw 'Bolt 7687 already in use; no process started.' }
catch [System.Net.Sockets.SocketException] { }
finally { $probe.Dispose() }
$classPath = "$(Join-Path $runtime 'build/classes');$(Join-Path $runtime 'lib/*')"
$dataHome = Join-Path $runtime 'data-home'
$socketTemp = Join-Path $runtime 'socket-temp'
New-Item -ItemType Directory -Force -Path $socketTemp | Out-Null
$process = Start-Process -FilePath $java -ArgumentList @('-Xms256m','-Xmx1g',"-Djdk.net.unixdomain.tmpdir=$socketTemp",'-cp',"`"$classPath`"",'LocalNeo4j',"`"$dataHome`"") -WorkingDirectory $resolvedRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'server.stdout.log') -RedirectStandardError (Join-Path $runtime 'server.stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $runtime 'server.pid')
Write-Output "Neo4j launched on loopback Bolt 7687; PID $($process.Id)."
