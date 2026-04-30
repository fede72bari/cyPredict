# Public API

This project keeps the notebook-compatible legacy API while adding structured
configuration objects for gradual migration.

## Supported Legacy Surface

- `cyPredict.cyPredict(...)`
- `cyPredict.cyPredict(...).analyze_and_plot(...)`
- `cyPredict.cyPredict(...).multiperiod_analysis(...)`
- `cyPredict.cyPredict(...).get_min_max_analysis_df(...)`
- `cyPredict.cyPredict(...).get_most_updated_optimization_pars(...)`

Legacy signatures remain supported for the current notebooks. Removed
parameters are listed in `docs/parameter_matrix.md`.

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
from cyPredict import AnalysisConfig, DetrendConfig, GoertzelConfig, OutputConfig

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
