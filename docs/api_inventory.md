# cyPredict API Inventory

Baseline inspected: `873ad825a4e4edbe4962b13362d56bcb68da5408`

This inventory is intentionally descriptive. It does not authorize removals by itself; every signature or return-contract change still requires a golden test comparison.

## Public Surface Observed In Notebooks

| Function | Observed use | Current role | Cleanup action |
| --- | --- | --- | --- |
| `cyPredict(...)` | frequent | data loading and state initialization | keep public; document data source modes |
| `analyze_and_plot(...)` | frequent | single period/range dominant-cycle analysis | keep public; separate calculation from plotting |
| `multiperiod_analysis(...)` | frequent | multirange analysis and signal reconstruction | keep public; introduce config object later |
| `get_most_updated_optimization_pars(...)` | frequent | choose most relevant optimization parameters by date | keep public; document dataframe contract |
| `get_min_max_analysis_df(...)` | frequent | incremental min/max analysis dataframe/CSV workflow | keep public; make worker-safe and idempotent |
| `min_max_analysis_concatenated_dataframe(...)` | indirect/frequent | row-level feature extraction around CDC projections | keep public or semi-public during transition |

## Function Map

| Line | Function | Docstring | Return count | Initial classification |
| ---: | --- | :---: | ---: | --- |
| 125 | `__init__` | no | 0 | public |
| 203 | `track_time` | no | 0 | internal logging/timing |
| 210 | `set_start_time` | no | 0 | internal logging/timing |
| 215 | `download_finance_data` | no | 0 | public/helper |
| 275 | `hp_filter` | no | 3 | calculation |
| 368 | `get_bartels_score` | no | 1 | calculation |
| 426 | `jh_filter` | no | 1 | calculation |
| 733 | `find_next_valid_datetime` | no | 2 | datetime helper |
| 776 | `datetime_dateset_extend` | no | 1 | datetime helper |
| 937 | `linear_detrend` | no | 1 | calculation |
| 964 | `analyze_and_plot` | no | 7 | public |
| 2058 | `multiperiod_analysis` | no | 6 | public |
| 3452 | `debug_check_complex_col` | yes | 1 | debug |
| 3470 | `debug_check_complex_values` | yes | 0 | debug |
| 3477 | `get_goertzel_amplitudes` | no | 1 | helper |
| 3482 | `cicles_composite_signals` | no | 1 | calculation |
| 3511 | `indict_MACD_SGMACD` | no | 1 | indicator helper |
| 3554 | `indict_RSI_SG_smooth_RSI` | no | 1 | indicator helper |
| 3584 | `custom_crossover` | no | 2 | optimization helper |
| 3596 | `indict_centered_average_deltas` | no | 1 | indicator helper |
| 3621 | `rebuilt_signal_zeros` | no | 1 | signal helper |
| 3783 | `get_row_score` | no | 1 | scoring helper |
| 3791 | `get_gloabl_score` | no | 1 | scoring helper |
| 3812 | `trade_predicted_dominant_cicles_peaks` | no | 4 | legacy/public candidate |
| 3884 | `CDC_vs_detrended_correlation` | no | 1 | optimization/scoring |
| 3960 | `CDC_vs_detrended_correlation_sum` | no | 1 | optimization/scoring |
| 4040 | `trade_predicted_dominant_cicles_peaks_sum` | no | 1 | legacy/scoring |
| 4102 | `genOpt_initializeIndividual` | no | 4 | optimization helper |
| 4167 | `discretized_uniform` | no | 1 | optimization helper |
| 4173 | `MultiAn_initializeIndividual` | no | 1 | optimization helper |
| 4201 | `MultiAn_evaluateFitness` | yes | 2 | calculation/native bridge |
| 4381 | `decode_individual` | no | 1 | optimization helper |
| 4406 | `MultiAn_optimize_NLOPT` | no | 2 | optimization helper |
| 4506 | `MultiAn_evaluateFitness_py` | no | 2 | calculation |
| 4587 | `MultiAn_cyclesAlignKPI` | no | 1 | KPI calculation |
| 4783 | `genOpt_evaluateMSEFitness` | no | 13 | optimization fitness |
| 4915 | `genOpt_evaluateFitness` | no | 9 | optimization fitness |
| 4990 | `genOpt_cycleParsGenOptimization` | yes | 1 | public/optimization |
| 5503 | `save_dataframe` | no | 1 | I/O helper |
| 5573 | `min_max_analysis` | no | 1 | feature extraction |
| 5640 | `min_max_analysis_concatenated_dataframe` | no | 1 | public/semi-public |
| 5824 | `get_min_max_analysis_df` | no | 2 | public |
| 5997 | `get_most_updated_optimization_pars` | no | 1 | public |
| 6107 | `detrend_lowess` | yes | 1 | calculation |

## Immediate Observations

- The module mixes public API, plotting, file I/O, optimization, native bridges, debug code, and report generation in one file.
- There are only a few existing docstrings; most functions need structured documentation.
- Several return tuples contain placeholders or `None` values that should be documented before any contract change.
- Long public signatures should be stabilized through config objects rather than repeated positional/keyword expansion.
- Notebook compatibility matters because historical notebooks still call older parameters such as `CDC_bb_analysis`, `CDC_RSI_analysis`, `CDC_MACD_analysis`, and sometimes `time_zone`.

