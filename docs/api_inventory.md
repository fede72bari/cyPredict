# cyPredict API Inventory

Current audit status: Milestone 2 API/parameter inventory closed for the
current modular layout under `cyPredict/core/`.

This inventory is intentionally descriptive. It does not authorize removals by itself; every signature or return-contract change still requires a golden test comparison.

## Public Surface Observed In Notebooks

| Function | Observed use | Current role | Cleanup action |
| --- | --- | --- | --- |
| `cyPredict(...)` | frequent | data loading and state initialization | keep public; document data source modes |
| `analyze_and_plot(...)` | frequent | single period/range dominant-cycle analysis | keep public; config and result bridges available |
| `multiperiod_analysis(...)` | frequent | multirange analysis and signal reconstruction | keep public; config and result bridges available |
| `get_most_updated_optimization_pars(...)` | frequent | choose most relevant optimization parameters by date | keep public; document dataframe contract |
| `get_min_max_analysis_df(...)` | frequent | incremental min/max analysis dataframe/CSV workflow | keep public; make worker-safe and idempotent |
| `min_max_analysis_concatenated_dataframe(...)` | indirect/frequent | row-level feature extraction around CDC projections | keep public or semi-public during transition |

## Current Function Map

Parameter/default details are tracked by the inspected signatures and guarded
by `tests/test_api_parameter_contracts.py`; removed and mode-specific
parameters are listed in `docs/parameter_matrix.md`.

| Module | Line | Function | Current role | Main side effects |
| --- | ---: | --- | --- | --- |
| `analysis.py` | 25 | `analyze_and_plot` | public single-range calculation | logging, optional plotting |
| `data.py` | 10 | `download_finance_data` | data loading | self data/state, file/network, logging |
| `dates.py` | 10 | `find_next_valid_datetime` | datetime helper | none |
| `dates.py` | 70 | `datetime_dateset_extend` | datetime helper | logging |
| `detrending.py` | 12 | `hp_filter` | calculation helper | logging |
| `detrending.py` | 119 | `jh_filter` | calculation helper | logging |
| `detrending.py` | 160 | `linear_detrend` | calculation helper | none |
| `detrending.py` | 195 | `detrend_lowess` | calculation helper | none |
| `diagnostics.py` | 7 | `debug_check_complex_col` | debug helper | logging |
| `diagnostics.py` | 33 | `debug_check_complex_values` | debug helper | none |
| `diagnostics.py` | 39 | `get_goertzel_amplitudes` | diagnostic getter | none |
| `extrema.py` | 10 | `trade_predicted_dominant_cicles_peaks` | legacy scoring helper | calls analysis |
| `extrema.py` | 91 | `CDC_vs_detrended_correlation` | optimization/scoring helper | calls multirange analysis, logging |
| `extrema.py` | 145 | `CDC_vs_detrended_correlation_sum` | optimization/scoring loop | logging |
| `extrema.py` | 230 | `trade_predicted_dominant_cicles_peaks_sum` | legacy scoring loop | calls analysis |
| `extrema.py` | 284 | `MultiAn_cyclesAlignKPI` | alignment KPI calculation | none |
| `indicators.py` | 12 | `indict_MACD_SGMACD` | indicator helper | mutates/returns dataframe |
| `indicators.py` | 72 | `indict_RSI_SG_smooth_RSI` | indicator helper | mutates/returns dataframe |
| `indicators.py` | 117 | `indict_centered_average_deltas` | indicator helper | mutates/returns dataframe |
| `minmax.py` | 13 | `min_max_analysis` | feature extraction | none |
| `minmax.py` | 97 | `min_max_analysis_concatenated_dataframe` | public/semi-public feature workflow | calls multirange analysis |
| `minmax.py` | 286 | `get_min_max_analysis_df` | public/semi-public CSV workflow | file I/O, logging |
| `multiperiod.py` | 37 | `multiperiod_analysis` | public multirange calculation | self state, multiprocessing, logging, optional plotting |
| `optimization.py` | 20 | `custom_crossover` | DEAP helper | mutates individuals |
| `optimization.py` | 43 | `genOpt_initializeIndividual` | optimization helper | reads optimizer state |
| `optimization.py` | 99 | `discretized_uniform` | optimization helper | random sampling |
| `optimization.py` | 105 | `MultiAn_initializeIndividual` | optimization helper | random sampling |
| `optimization.py` | 133 | `MultiAn_evaluateFitness` | fitness/native bridge | native fitness call |
| `optimization.py` | 205 | `decode_individual` | optimization helper | none |
| `optimization.py` | 230 | `MultiAn_optimize_NLOPT` | NLopt helper | logging |
| `optimization.py` | 334 | `MultiAn_evaluateFitness_py` | Python fitness helper | logging |
| `optimization.py` | 418 | `genOpt_evaluateMSEFitness` | optimizer fitness | logging, calls multirange analysis |
| `optimization.py` | 539 | `genOpt_evaluateFitness` | optimizer fitness | logging, calls scoring helpers |
| `optimization.py` | 619 | `genOpt_cycleParsGenOptimization` | public optimization workflow | self optimizer state, multiprocessing, file I/O, logging |
| `persistence.py` | 13 | `save_dataframe` | I/O helper | CSV read/write |
| `persistence.py` | 106 | `get_most_updated_optimization_pars` | I/O helper | CSV read, logging |
| `plotting.py` | 20 | `plot_single_range_analysis_charts` | notebook plotting helper | Plotly display, logging |
| `plotting.py` | 136 | `plot_multiperiod_analysis_charts` | notebook plotting helper | Plotly display, HTML output, logging |
| `reconstruction.py` | 10 | `cicles_composite_signals` | signal reconstruction helper | none |
| `reconstruction.py` | 62 | `rebuilt_signal_zeros` | signal helper | none |
| `scoring.py` | 9 | `get_row_score` | scoring helper | none |
| `scoring.py` | 15 | `get_gloabl_score` | scoring helper | none |
| `spectral.py` | 9 | `get_bartels_score` | spectral helper | none |
| `state.py` | 23 | `__init__` | public lifecycle/state initializer | data load, self state, logger |
| `state.py` | 169 | `is_log_enabled` | logging helper | none |
| `state.py` | 172 | `configure_logging` | logging helper | logger replacement |
| `state.py` | 192 | `log_debug` | logging helper | structured logging |
| `state.py` | 195 | `log_info` | logging helper | structured logging |
| `state.py` | 198 | `log_warning` | logging helper | structured logging |
| `state.py` | 201 | `log_error` | logging helper | structured logging |
| `state.py` | 204 | `log_timing` | logging helper | structured logging |

## Immediate Observations

- The original monolithic module has been split into core mixins, but the public class remains `cyPredict.cyPredict`.
- Most core functions now have docstrings; exhaustive docstring cleanup remains Milestone 6.
- Several return tuples contain placeholders or `None` values that should be documented before any contract change.
- Long public signatures now have transitional config objects; legacy signatures stay supported for notebooks.
- Notebook compatibility matters; removed legacy parameters are tracked in `docs/parameter_matrix.md` and guarded by tests.
- Structured logging utilities are wired into the main analysis paths. Legacy `time_tracking` and `print_activity_remarks` flags are removed.
- Result wrappers are additive; legacy tuple/dataframe returns are unchanged.
