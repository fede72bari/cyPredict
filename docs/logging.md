# Logging Design

`cyPredict.logging_utils` introduces a lightweight structured logger that can be adopted progressively without changing calculation paths.

Each event includes:

- UTC timestamp;
- run id;
- log level;
- category;
- function name;
- elapsed seconds from logger creation;
- ticker;
- timeframe;
- message;
- context dictionary.

Supported categories:

- `processing`
- `debug`
- `timing`
- `calculation`
- `data`
- `io`
- `optimization`
- `warning`
- `error`

Example:

```python
from cyPredict.logging_utils import CyPredictLogger

logger = CyPredictLogger(
    ticker="ES=F",
    timeframe="5m",
    log_to_console=True,
    log_to_file=True,
)

logger.timing("Goertzel transform completed", function="analyze_and_plot", periods=512)
```

File outputs use structured names:

```text
logs/cypredict_{ticker}_{timeframe}_{YYYYMMDD_HHMMSS}_{run_id}.log
logs/cypredict_{ticker}_{timeframe}_{YYYYMMDD_HHMMSS}_{run_id}.jsonl
```

Runtime controls:

- `log_level="WARNING"` keeps normal notebook runs quiet except warnings and errors.
- `log_level="INFO"` enables processing and timing events.
- `log_level="DEBUG"` enables diagnostic context and intermediate display blocks.
- `log_to_console=True` prints structured lines to console.
- `log_to_file=True` writes both `.log` and `.jsonl` files.

Legacy flags `time_tracking` and `print_activity_remarks` were removed from
the public API. Timing is now category `timing`; verbose remarks are now
controlled by `log_level`.
