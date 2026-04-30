# Native And Custom Dependencies

Baseline inspected: `a96ec0690052bca1fa36f80064ec47997de0dc8b`

`cyPredict/cypredict.py` currently imports several modules that were historically resolved by notebooks adding the sibling library root to `sys.path`:

```python
sys.path.append(r"D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES")
```

The source files have now been copied into `native/` so they can be versioned and rebuilt from the project.

## Python Package With Native Component

| Import | PyPI package | Notes |
| --- | --- | --- |
| `talib` | `TA-Lib` | May require a compatible wheel or local TA-Lib native library depending on Python/platform. |

## Custom Native Modules

| Import | Observed source location | Current status |
| --- | --- | --- |
| `goertzel` | `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\GOERTZEL TRANSFORM C\V1.2` | `native/goertzel` |
| `cyfitness` | `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyFitness` | `native/cyfitness` |
| `cyGAopt` | `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyGAopt` | `native/cygaopt` |
| `cyGAoptMultiCore` | `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyGAoptMulticore` | `native/cygaopt_multicore` |
| `genetic_optimization` | `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\GeneticOptimization` | `native/genetic_optimization_legacy`; legacy/commented in current import block |

## Target Structure

The planned cleanup should move source and build scripts under:

```text
native/
  goertzel/
  cyfitness/
  cygaopt/
  cygaopt_multicore/
  genetic_optimization_legacy/
```

Build artifacts such as `.pyd`, `.obj`, `.lib`, CMake output and Visual Studio output should be generated locally or attached to releases, not committed as normal source files unless there is an explicit release-artifact policy.

Use `scripts/build_native.ps1` from the repository root to build with the Anaconda `cyenv` interpreter.

The build script now compiles all native modules, including `cyGAopt` and
`cyGAoptMultiCore`, using setuptools/pybind11. It initializes the local Visual
Studio 2022 C++ toolchain when `cl.exe` is not already on `PATH`.

By default it writes extensions under each module's `build/lib.*` folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_native.ps1
```

Use `-InPlace` only after closing notebooks or Python processes that may have
loaded the old `.pyd` files:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_native.ps1 -InPlace
```

`cyPredict/cypredict.py` now prepends native `build/lib.*` folders first and
then the `native/*` source folders to `sys.path` before importing the custom
modules. `cyPredict/__init__.py` remains a compatibility re-export for legacy
imports. This makes locally built `.pyd` files discoverable without requiring
the notebook working directory and without replacing a locked in-place module.

`cyGAopt` and `cyGAoptMultiCore` expose `ABI_VERSION = 2`. Python import guards
reject stale versions before the C++ GA branch can run.
