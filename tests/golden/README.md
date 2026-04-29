# Golden Test Data

This directory is reserved for reproducible golden baseline artifacts.

Recommended layout:

```text
tests/golden/
  scenarios/
    <scenario>.json
  baselines/
    <scenario>.json
```

Generate a baseline with the Anaconda `cyenv` interpreter:

```powershell
& "C:\Users\Federico\anaconda3\envs\cyenv\python.exe" scripts\capture_golden_baseline.py `
  --config tests\golden\scenarios\<scenario>.json `
  --output-dir tests\golden\baselines
```

Prefer file-based OHLCV fixtures when a frozen input file is available. The first committed scenario, `qqq_eod_analyze_and_plot`, intentionally uses `yfinance` EOD data on a closed historical range to match the notebook workflow.
Because Yahoo data can drift slightly between downloads, that comparison uses numeric tolerances and does not require exact full-data hashes.

Run the golden comparison test explicitly:

```powershell
$env:CYPREDICT_RUN_GOLDEN='1'
& "C:\Users\Federico\anaconda3\envs\cyenv\python.exe" -m pytest tests\test_golden_cypredict.py -q
```
