"""Native extension import helpers for cyPredict."""

from __future__ import annotations

import sys
from pathlib import Path


REQUIRED_CYGAOPT_ABI_VERSION = 2

NATIVE_SOURCE_DIRS = [
    Path(__file__).resolve().parents[1] / "native" / "goertzel",
    Path(__file__).resolve().parents[1] / "native" / "cyfitness",
    Path(__file__).resolve().parents[1] / "native" / "cygaopt",
    Path(__file__).resolve().parents[1] / "native" / "cygaopt_multicore",
    Path(__file__).resolve().parents[1] / "native" / "genetic_optimization_legacy",
]


def native_module_dirs() -> list[Path]:
    """Return build-output folders first, then source folders with in-place builds."""

    build_dirs: list[Path] = []
    for native_source_dir in NATIVE_SOURCE_DIRS:
        build_root = native_source_dir / "build"
        if build_root.exists():
            build_dirs.extend(sorted(build_root.glob("lib.*"), reverse=True))
    return build_dirs + NATIVE_SOURCE_DIRS


def ensure_native_module_paths() -> None:
    """Prepend repository native-extension folders to ``sys.path``."""

    for native_module_dir in reversed(native_module_dirs()):
        if native_module_dir.exists():
            native_module_path = str(native_module_dir)
            if native_module_path not in sys.path:
                sys.path.insert(0, native_module_path)


def require_native_abi(module, module_name: str, expected_abi: int) -> None:
    """Raise a clear error when a stale native extension is imported."""

    actual_abi = getattr(module, "ABI_VERSION", None)
    if actual_abi != expected_abi:
        module_file = getattr(module, "__file__", "<unknown>")
        raise ImportError(
            f"{module_name} ABI_VERSION={actual_abi!r} loaded from {module_file}; "
            f"expected ABI_VERSION={expected_abi}. Rebuild native extensions with "
            "scripts/build_native.ps1 using the cyenv interpreter."
        )
