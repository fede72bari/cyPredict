# Public API

This project keeps the notebook-compatible legacy API while adding structured
configuration objects for gradual migration.

## Supported Legacy Surface

- `cyPredict.cyPredict(...)`
- `cypredict.CyPredict(...)`
- `cyPredict.cyPredict(...).analyze_and_plot(...)`
- `cyPredict.cyPredict(...).multiperiod_analysis(...)`
- `cyPredict.cyPredict(...).get_min_max_analysis_df(...)`
- `cyPredict.cyPredict(...).get_most_updated_optimization_pars(...)`

Legacy signatures remain supported for the current notebooks. Removed
parameters are listed in `docs/parameter_matrix.md`.

The lowercase `cypredict` module is a compatibility facade for application code
that prefers conventional package naming. It aliases `CyPredict` to the same
class used by the historical `cyPredict` package; it does not introduce a
second implementation.

## Structured Config Objects

Config dataclasses live in `cyPredict.config` and are re-exported from the
package root:

- `DataConfig`
- `DetrendConfig`
- `GoertzelConfig`
- `OptimizationConfig`
- `ProjectionConfig`
- `OutputConfig`
- `AnalysisConfig`
- `MultiPeriodAnalysisConfig`

They do not change the calculation path. They expand into the same keyword
arguments used by the legacy methods.

## Single-Range Example

```python
from cypredict import CyPredict
from cyPredict import AnalysisConfig, DetrendConfig, GoertzelConfig, OutputConfig

cp = CyPredict(ticker="QQQ", data_start_date="2020-01-01", data_timeframe="1d")

config = AnalysisConfig(
    data_column_name="Close",
    current_date="2026-04-21",
    num_samples=512,
    goertzel=GoertzelConfig(min_period=32, max_period=64, kaiser_beta=1),
    detrend=DetrendConfig(detrend_type="hp_filter", hp_filter_lambda=16947),
    output=OutputConfig(show_charts=True, print_report=False),
)

result = cp.analyze_and_plot_from_config(config)
```

## Multiperiod Example

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
    ),
    goertzel=GoertzelConfig(windowing="kaiser", kaiser_beta=1),
    optimization=OptimizationConfig(
        population_n=100,
        CXPB=0.8,
        MUTPB=0.3,
        NGEN=4000,
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

## Naming Policy

Historical names with typos, such as `cicles`, `weigth`, `weigthed`, and
`gloabl`, are kept where changing them would break notebooks or stored output.
New config objects use clearer grouping but still translate to the legacy
keyword names required by the existing implementation.

## Result Objects

The legacy return tuples/dataframes are unchanged. Transitional result-object
wrappers are available when callers want named fields:

- `cp.analyze_and_plot_result(...)` -> `AnalysisResult`
- `cp.multiperiod_analysis_result(...)` -> `MultiPeriodResult`
- `cp.min_max_analysis_concatenated_dataframe_result(...)` -> `MinMaxAnalysisResult`
- `cp.get_min_max_analysis_df_result(...)` -> `MinMaxAnalysisResult`

Each result object can be converted back to the legacy contract with
`as_legacy_tuple()` or, for min/max dataframe workflows, `as_legacy_value()`.
