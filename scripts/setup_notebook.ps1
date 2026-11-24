param([string]$PythonExecutable = 'python', [switch]$RefreshDependencies)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentRoot = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $environmentRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & $PythonExecutable -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$lockFile = Join-Path $projectRoot 'requirements-lock.txt'
$requirementsFile = Join-Path $projectRoot 'requirements-dev.txt'
if ((Test-Path -LiteralPath $lockFile) -and -not $RefreshDependencies) {
    $requirementsFile = $lockFile
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    & $pythonPath -m ipykernel install --prefix $environmentRoot --name oil-inflation --display-name 'Python (Oil Inflation Project)'
    if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
    if (-not (Test-Path -LiteralPath $lockFile) -or $RefreshDependencies) {
        $packages = & $pythonPath -m pip freeze
        $packages | Set-Content -LiteralPath $lockFile -Encoding UTF8
    }
    Write-Output 'Ready. Start JupyterLab with .\scripts\start_jupyter.ps1'
}
finally { Pop-Location }
