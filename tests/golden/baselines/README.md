# Baseline JSON Artifacts

Captured baseline summaries are written here by `scripts/capture_golden_baseline.py`.

The JSON files are intended to be committed only when they are based on frozen input data and a known code/native-module baseline.

The `qqq_eod_analyze_and_plot.json` artifact is based on a closed historical `yfinance` EOD range, not a local CSV snapshot. Treat provider or `yfinance` version changes as possible causes if this artifact is recaptured and drifts.
