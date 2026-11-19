param([string]$PythonExe='python',[string]$OutputDirectory=('replay-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')))
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$run=(New-Item -ItemType Directory -Path $OutputDirectory).FullName
& $PythonExe -I -B .\work\benchmark\audit.py --out "$run/audit.json"
if($LASTEXITCODE){throw 'Audit failed'}
& $PythonExe -I -B .\work\benchmark\benchmark.py --blocks 6 --out "$run/counterbalanced"
if($LASTEXITCODE){throw 'Benchmark failed'}
