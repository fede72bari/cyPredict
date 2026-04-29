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

Do not capture Yahoo/live-provider data as a golden baseline unless the raw data snapshot is also stored or frozen. Prefer file-based OHLCV fixtures exported from the notebooks.

