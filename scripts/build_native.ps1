param(
    [string]$Python = "C:\Users\Federico\anaconda3\envs\cyenv\python.exe",
    [switch]$InPlace
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$repoRoot = Split-Path -Parent $PSScriptRoot

function Initialize-MsvcBuildEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    $installPath = $null
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -latest -products * -property installationPath
    }

    if (-not $installPath) {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\Community"
        if (Test-Path -LiteralPath $candidate) {
            $installPath = $candidate
        }
    }

    if (-not $installPath) {
        Write-Host "MSVC not found. Native builds requiring C/C++ compilation may fail."
        return
    }

    $msvcToolsRoot = Join-Path $installPath "VC\Tools\MSVC"
    $msvcTools = Get-ChildItem -LiteralPath $msvcToolsRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1

    if (-not $msvcTools) {
        Write-Host "MSVC tools folder not found under $msvcToolsRoot."
        return
    }

    $windowsKitsRoot = "C:\Program Files (x86)\Windows Kits\10"
    $sdkIncludeRoot = Join-Path $windowsKitsRoot "Include"
    $sdkVersion = Get-ChildItem -LiteralPath $sdkIncludeRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1

    if (-not $sdkVersion) {
        Write-Host "Windows SDK include folder not found under $sdkIncludeRoot."
        return
    }

    $msvcLib = Join-Path $msvcTools.FullName "lib\x64"
    if (-not (Test-Path -LiteralPath (Join-Path $msvcLib "msvcprt.lib"))) {
        $oneCoreLib = Join-Path $msvcTools.FullName "lib\onecore\x64"
        if (Test-Path -LiteralPath (Join-Path $oneCoreLib "msvcprt.lib")) {
            $msvcLib = $oneCoreLib
        }
    }

    $sdkName = $sdkVersion.Name
    $msvcBin = Join-Path $msvcTools.FullName "bin\Hostx64\x64"
    $sdkBin = Join-Path $windowsKitsRoot "bin\$sdkName\x64"
    $sdkFallbackBin = Join-Path $windowsKitsRoot "bin\x64"
    $sdkInclude = Join-Path $windowsKitsRoot "Include\$sdkName"
    $sdkLib = Join-Path $windowsKitsRoot "Lib\$sdkName"

    $env:PATH = "$msvcBin;$sdkBin;$sdkFallbackBin;$env:PATH"
    $env:INCLUDE = "$($msvcTools.FullName)\include;$sdkInclude\ucrt;$sdkInclude\shared;$sdkInclude\um;$sdkInclude\winrt;$sdkInclude\cppwinrt"
    $env:LIB = "$msvcLib;$sdkLib\ucrt\x64;$sdkLib\um\x64"
    $env:DISTUTILS_USE_SDK = "1"
    $env:MSSdk = "1"

    Write-Host "Initialized MSVC build environment: $($msvcTools.Name), SDK $sdkName"
}

Write-Host "Using Python: $Python"
& $Python --version

Initialize-MsvcBuildEnvironment

$setuptoolsBuilds = @(
    @{ Name = "goertzel"; Path = Join-Path $repoRoot "native\goertzel"; Setup = "setup.py" },
    @{ Name = "cyfitness"; Path = Join-Path $repoRoot "native\cyfitness"; Setup = "cyfitness_setup.py" },
    @{ Name = "cyGAopt"; Path = Join-Path $repoRoot "native\cygaopt"; Setup = "setup.py" },
    @{ Name = "cyGAoptMultiCore"; Path = Join-Path $repoRoot "native\cygaopt_multicore"; Setup = "setup.py" },
    @{ Name = "genetic_optimization_legacy"; Path = Join-Path $repoRoot "native\genetic_optimization_legacy"; Setup = "genetic_optimization_setup.py" }
)

foreach ($build in $setuptoolsBuilds) {
    Push-Location $build.Path
    try {
        Write-Host "Building $($build.Name)"
        $buildArgs = @($build.Setup, "build_ext", "--force")
        if ($InPlace) {
            $buildArgs += "--inplace"
        }
        & $Python @buildArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed for $($build.Name) with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
