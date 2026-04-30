"""Minimal cyPredict import example.

This example intentionally avoids constructing ``cyPredict.cyPredict`` so it
does not download data. It verifies that the legacy package import resolves and
that native extension paths are registered.
"""

from cyPredict import cyPredict
from cyPredict.native_imports import ensure_native_module_paths


def main() -> None:
    ensure_native_module_paths()
    print(f"Loaded callable legacy class: {cyPredict}")
    print(f"Loaded module-style legacy alias: {cyPredict.cyPredict}")


if __name__ == "__main__":
    main()
