param(
    [switch]$InPlaceExtensions
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$nativeRoot = Join-Path $repoRoot "native"
$nativeRootResolved = (Resolve-Path -LiteralPath $nativeRoot).Path

function Remove-NativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($nativeRootResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside native root: $resolved"
    }

    Write-Host "Removing $resolved"
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

Get-ChildItem -LiteralPath $nativeRoot -Directory | ForEach-Object {
    Remove-NativePath -Path (Join-Path $_.FullName "build")
}

if ($InPlaceExtensions) {
    Get-ChildItem -LiteralPath $nativeRoot -Directory | ForEach-Object {
        Get-ChildItem -LiteralPath $_.FullName -Filter "*.pyd" -File -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-NativePath -Path $_.FullName
        }
    }
}
