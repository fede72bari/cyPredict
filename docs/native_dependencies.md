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

`cyPredict/cypredict.py` now prepends these `native/*` folders to `sys.path` before importing the custom modules. `cyPredict/__init__.py` remains a compatibility re-export for legacy imports. This makes locally built or locally copied `.pyd` files discoverable without requiring the notebook working directory.
