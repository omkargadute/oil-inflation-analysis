param([int]$Port = 8888, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements-dev.txt first; see docs/NOTEBOOK_SETUP.md.'
}
$jupyterArgs = @('-m', 'jupyterlab', '--ip=127.0.0.1', "--port=$Port", "--ServerApp.root_dir=$projectRoot", 'notebooks/final_project.ipynb')
if ($NoBrowser) { $jupyterArgs += '--no-browser' }
Push-Location -LiteralPath $projectRoot
try { & $pythonPath @jupyterArgs }
finally { Pop-Location }
