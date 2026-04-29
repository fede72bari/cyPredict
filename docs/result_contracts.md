# Result Contracts - Initial Notes

Baseline inspected: `873ad825a4e4edbe4962b13362d56bcb68da5408`

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

## Target Result Objects

Suggested future result classes:

- `AnalysisResult`
- `MultiPeriodResult`
- `MinMaxAnalysisResult`
- `ProjectionResult`

Each should provide:

- named attributes;
- `as_legacy_tuple()` for notebook compatibility;
- `to_dict()` for worker/API use;
- metadata with code version, config hash, and warnings.

