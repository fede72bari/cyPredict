# Result Contracts

Current milestone status: Milestone 4 transitional result objects are
available without changing legacy returns.

## Existing Tuple Returns

### `analyze_and_plot(...)`

Successful return:

```python
(
    current_date,
    index_of_max_time_for_cd,
    original_data,
    signals_results,
    configuration,
)
```

Failure/invalid input paths currently return five `None` values in several branches. This should be documented and eventually replaced by structured errors or a result object with `ok=False`.

### `multiperiod_analysis(...)`

Successful return:

```python
(
    elaborated_data_series,
    signals_results_series,
    composite_signal,
    configurations_series,
    None,
    None,
    index_of_max_time_for_cd,
    scaled_signals,
    best_fitness_value,
)
```

The two `None` positions appear to be legacy placeholders. Before removal, confirm whether downstream notebooks unpack them.

### `min_max_analysis_concatenated_dataframe(...)`

Returns one dataframe row combining:

- analysis datetime;
- best fitness;
- OHLCV-derived values;
- scaled detrended signal;
- min/max features for price, Goertzel signal, alignment KPI, and weighted alignment KPI.

### `get_min_max_analysis_df(...)`

Returns the accumulated dataframe and writes it to CSV during processing.

## Result Objects

Implemented result classes live in `cyPredict.results` and are re-exported
from the package root:

- `AnalysisResult`
- `MultiPeriodResult`
- `MinMaxAnalysisResult`

Each provides named attributes, conversion back to the legacy return contract,
and a compact `to_dict()` for worker/API-oriented code.

### `AnalysisResult`

Created by:

```python
result = cp.analyze_and_plot_result(...)
```

Legacy conversion:

```python
current_date, index_of_max_time_for_cd, original_data, signals_results, configuration = result.as_legacy_tuple()
```

### `MultiPeriodResult`

Created by:

```python
result = cp.multiperiod_analysis_result(...)
```

Legacy conversion:

```python
(
    elaborated_data_series,
    signals_results_series,
    composite_signal,
    configurations_series,
    bb_delta,
    cdc_rsi,
    index_of_max_time_for_cd,
    scaled_signals,
    best_fitness_value,
) = result.as_legacy_tuple()
```

The object also preserves the shorter six-value failure tuple emitted by a
legacy early-return path, so wrapper use does not hide that behavior.

### `MinMaxAnalysisResult`

Created by:

```python
result = cp.min_max_analysis_concatenated_dataframe_result(...)
result = cp.get_min_max_analysis_df_result(...)
```

Legacy conversion:

```python
dataframe = result.as_legacy_value()
```

## Still Deferred

- `ProjectionResult` remains a future GammaSignalForge-facing object.
- Code-version metadata and config hashes are deferred until config/result
  contracts are both stable.
