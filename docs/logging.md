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

Adoption order:

1. wire the logger into initialization while keeping existing `print_activity_remarks` behavior;
2. replace timing prints;
3. replace processing/debug prints;
4. keep notebook console output controlled by log level.

