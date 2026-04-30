param(
    [string]$PythonExe = "C:\Users\Federico\anaconda3\envs\cyenv\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $repoRoot
try {
    & $PythonExe -m pip install -e .
    & $PythonExe -c "from cyPredict import cyPredict; import importlib.metadata as md; print(md.version('cypredict')); print(cyPredict); print(cyPredict.cyPredict)"
}
finally {
    Pop-Location
}
