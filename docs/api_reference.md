# API Reference

This reference summarizes the maintained public surface after the first
cleanup/refactor milestones. Detailed parameter behavior lives in source
docstrings; `tests/test_docstring_contracts.py` enforces that maintained core
functions and classes keep non-empty docstrings.

## Package Imports

```python
from cyPredict import cyPredict
from cypredict import CyPredict
from cyPredict import AnalysisConfig, MultiPeriodAnalysisConfig
from cyPredict import AnalysisResult, MultiPeriodResult, MinMaxAnalysisResult
```

The legacy notebook pattern remains supported:

```python
from cyPredict import cyPredict

cp = cyPredict.cyPredict(
    data_source="yfinance",
    ticker="QQQ",
    data_start_date="2020-01-01",
    data_end_date="2024-01-01",
    data_timeframe="1d",
)
```

Direct class construction is also supported:

```python
cp = cyPredict(
    data_source="yfinance",
    ticker="QQQ",
    data_start_date="2020-01-01",
    data_end_date="2024-01-01",
    data_timeframe="1d",
)
```

Application code can use the lowercase facade:

```python
from cypredict import CyPredict

cp = CyPredict(
    data_source="yfinance",
    ticker="QQQ",
    data_start_date="2020-01-01",
    data_end_date="2024-01-01",
    data_timeframe="1d",
)
```

## Legacy Analysis Methods

### `analyze_and_plot(...)`

Runs the single-range Goertzel cycle analysis and optional notebook plotting.
It returns the historical five-value tuple:

```python
current_date, index_of_max_time_for_cd, original_data, signals_results, configuration = (
    cp.analyze_and_plot(...)
)
```

Use `analyze_and_plot_result(...)` when named fields are preferable:

```python
result = cp.analyze_and_plot_result(...)
configuration = result.configuration
```

### `multiperiod_analysis(...)`

Runs the multi-range dominant-cycle workflow, including detrending, Goertzel
extraction, amplitude/frequency/phase optimization and signal reconstruction.
It keeps the historical nine-value success tuple:

```python
(
    elaborated_data_df,
    signals_results_df,
    composite_signal,
    configurations,
    bb_delta,
    cdc_rsi,
    index_of_max_time_for_cd,
    scaled_signals,
    best_fitness_value,
) = cp.multiperiod_analysis(...)
```

Use `multiperiod_analysis_result(...)` for named fields while preserving the
legacy tuple through `as_legacy_tuple()`.

## Structured Config Methods

`AnalysisConfig` and `MultiPeriodAnalysisConfig` group related parameters
without changing the underlying calculation path. They flatten to the same
keyword arguments used by the legacy methods.

```python
from cyPredict import (
    DetrendConfig,
    GoertzelConfig,
    MultiPeriodAnalysisConfig,
    OptimizationConfig,
    OutputConfig,
    ProjectionConfig,
)

config = MultiPeriodAnalysisConfig(
    data_column_name="Close",
    current_date="2026-04-21",
    periods_pars=cicles_parameters_restr,
    detrend=DetrendConfig(
        detrend_type="hp_filter",
        linear_filter_window_size_multiplier=1.85,
        lowess_k=6,
    ),
    goertzel=GoertzelConfig(windowing="kaiser", kaiser_beta=1),
    optimization=OptimizationConfig(
        population_n=100,
        CXPB=0.8,
        MUTPB=0.3,
        NGEN=4000,
        MultiAn_fitness_type="mse",
        opt_algo_type="genetic_omny_frequencies",
        amplitudes_inizialization_type="all_equal_middle_value",
        frequencies_ft=True,
        phases_ft=True,
    ),
    projection=ProjectionConfig(
        reference_detrended_data="less_detrended",
        period_related_rebuild_range=True,
        period_related_rebuild_multiplier=1.5,
    ),
    output=OutputConfig(show_charts=True, log_level="INFO"),
)

result = cp.multiperiod_analysis_from_config(config)
```

## Conditional Parameters

- `windowing`: when `None`, no window is applied before the transform; when
  `"kaiser"`, `kaiser_beta` controls the Kaiser window shape.
- `lowess_k`: meaningful only for LOWESS detrending paths.
- `linear_filter_window_size_multiplier`: meaningful for linear detrending
  paths and related optimization domains.
- `period_related_rebuild_multiplier`: meaningful only when
  `period_related_rebuild_range` is true.
- `frequencies_ft` and `phases_ft`: control whether optimization vectors
  include frequency and phase genes. Disabled sections stay fixed to Goertzel
  peak values.
- `enabled_multiprocessing`: meaningful only for algorithms that expose a
  multiprocessing or multicore branch.

## Logging

All new console/file diagnostics go through `CyPredictLogger`. Runtime methods
can set:

```python
cp.configure_logging(log_level="INFO", log_to_console=True, log_to_file=False)
```

or pass logging keywords to `multiperiod_analysis(...)`:

```python
cp.multiperiod_analysis(..., log_level="INFO", log_to_console=True)
```

`INFO` includes processing and timing events; `DEBUG` adds diagnostic context.
