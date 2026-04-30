"""Minimal cyPredict import example.

This example intentionally avoids constructing ``cyPredict.cyPredict`` so it
does not download data. It verifies that the legacy package import resolves and
that native extension paths are registered. It also verifies the lowercase
``cypredict.CyPredict`` facade used by application code.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cyPredict import cyPredict
from cyPredict.native_imports import ensure_native_module_paths
from cypredict import CyPredict


def main() -> None:
    ensure_native_module_paths()
    print(f"Loaded callable legacy class: {cyPredict}")
    print(f"Loaded module-style legacy alias: {cyPredict.cyPredict}")
    print(f"Loaded lowercase facade class: {CyPredict}")


if __name__ == "__main__":
    main()
