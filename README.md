# cyPredict

`cyPredict` is a research-oriented Python package for dominant-cycle analysis on financial time series. The current implementation centers on the `cyPredict.cyPredict` class, which combines detrending, Goertzel transforms, dominant-period selection, signal reconstruction, genetic/NLopt-style optimization, and future local-min/max projection workflows.

The code is currently being consolidated. The first rule of the cleanup is to preserve numerical behavior against golden baseline scenarios before removing parameters, moving calculation code, or changing return contracts.

## Current Status

- Legacy package interface: `cyPredict/__init__.py`
- Current implementation module: `cyPredict/cypredict.py`
- Planned cleanup: `sviluppi_programmati/cypredict_cleanup_refactor_plan.md`
- Native dependency notes: `docs/native_dependencies.md`
- Native/custom dependencies are versioned under `native/` and loaded from local build output when available: `goertzel`, `cyfitness`, `cyGAopt`, `cyGAoptMultiCore`
- Research notebooks and usage examples currently live outside this repository under `D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS`

## Installation For Local Development

```powershell
python -m pip install -e .
```

The editable install only packages the Python code in this repository. Build native modules with the project script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_native.ps1
```

The default output is `native/*/build/lib.*`, which cyPredict prefers over older in-place `.pyd` files.

## Minimal Import

```python
import cyPredict

model = cyPredict.cyPredict(
    data_source="yfinance",
    ticker="SPY",
    data_start_date="2020-01-01",
    data_timeframe="1d",
)
```

For a no-download import check, run:

```powershell
python .\examples\minimal_import.py
```

Packaging can be verified with the project script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_packaging.ps1
```

## Cleanup Priorities

1. Establish golden baseline tests for daily, intraday, file-based, and multirange scenarios.
2. Document the public API, parameter interactions, return values, and side effects.
3. Centralize logging and remove uncontrolled console output.
4. Move custom native sources/build artifacts into a reproducible project structure.
5. Add a worker-friendly projection API for periodic use by GammaSignalForge.

## Disclaimer

This project is research software for time-series analysis. It is not financial advice and does not provide trade recommendations by itself.
