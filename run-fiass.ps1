# Runs fiass.py using the .venv-faiss virtual environment
$venvPython = Join-Path $PSScriptRoot ".venv-faiss\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "fiass.py"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at $venvPython"
    exit 1
}

& $venvPython $script
exit $LASTEXITCODE
