param(
    [string]$Python = "C:\Users\Federico\anaconda3\envs\cyenv\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Using Python: $Python"
& $Python --version

$setuptoolsBuilds = @(
    @{ Name = "goertzel"; Path = Join-Path $repoRoot "native\goertzel"; Setup = "setup.py" },
    @{ Name = "cyfitness"; Path = Join-Path $repoRoot "native\cyfitness"; Setup = "cyfitness_setup.py" },
    @{ Name = "genetic_optimization_legacy"; Path = Join-Path $repoRoot "native\genetic_optimization_legacy"; Setup = "genetic_optimization_setup.py" }
)

foreach ($build in $setuptoolsBuilds) {
    Push-Location $build.Path
    try {
        Write-Host "Building $($build.Name)"
        & $Python $build.Setup build_ext --inplace
    }
    finally {
        Pop-Location
    }
}

Write-Host "CMake-based modules are source-controlled in native\cygaopt and native\cygaopt_multicore."
Write-Host "Build those with the local Visual Studio/CMake workflow after confirming compiler configuration."

