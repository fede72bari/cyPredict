# Scenario JSON Files

Scenario files drive `scripts/capture_golden_baseline.py`.

Minimal shape:

```json
{
  "name": "example_file_analyze_and_plot",
  "constructor": {
    "data_source": "file",
    "data_filename": "path/to/frozen_ohlcv.csv",
    "data_timeframe": "5m",
    "print_activity_remarks": false,
    "time_tracking": false
  },
  "method": "analyze_and_plot",
  "method_kwargs": {
    "data_column_name": "Close",
    "num_samples": 400,
    "current_date": "2024-06-24T23:50:00+00:00",
    "final_kept_n_dominant_circles": 4,
    "min_period": 8,
    "max_period": 256,
    "detrend_type": "hp_filter",
    "show_charts": false,
    "print_report": false,
    "time_tracking": false
  }
}
```

Use scenarios extracted from the notebooks under:

```text
D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS
```

Committed scenarios:

- `qqq_eod_analyze_and_plot.json`: downloads QQQ daily data from `yfinance` over the closed range `2022-01-01` to `2024-01-01` and captures `analyze_and_plot` at `2023-12-29`.
