# Parameter Matrix

Current milestone status: Milestone 2 closed after AST unused-parameter audit,
notebook/call-site review, and regression tests.

Status labels:

- `active`: directly used in calculation or control flow.
- `routing`: passed to another function or stored for later use.
- `mode-specific`: meaningful only for selected algorithms or options.
- `legacy`: retained for older callers or notebooks.
- `unused-candidate`: not observed in direct body usage during initial static scan; requires confirmation.
- `remove-approved`: may be removed only after notebook scan and golden tests.
- `removed`: removed after local call-site verification and test coverage.

## Conditional Parameter Rules

| Parameter | Meaningful when | Notes |
| --- | --- | --- |
| `kaiser_beta` | `windowing == "kaiser"` | Should be ignored or validated otherwise. |
| `lowess_k` | `detrend_type == "lowess"` | Controls LOWESS window factor. |
| `linear_filter_window_size_multiplier` | linear detrend paths or optimization ranges that use it | Not meaningful for pure HP filter paths unless explicitly routed. |
| `period_related_rebuild_multiplier` | `period_related_rebuild_range == True` | Should be documented as inactive when range limiting is off. |
| `frequencies_ft` | algorithms optimizing frequencies | Inactive for amplitude-only or fixed-frequency paths. |
| `phases_ft` | algorithms optimizing phases | Inactive when phase is taken from Goertzel output. |
| `show_charts` | plotting/reporting paths | Must not affect numerical calculation. |
| `enabled_multiprocessing` | algorithms with parallel branch | Should not change deterministic output except ordering/timing. |

## Current Unused-Parameter Audit

The current AST scan over `cyPredict/core/*.py` has no unreviewed unused
parameters. The only allowed unused parameter is `grad` in the nested NLopt
callback `optimization.py::loss`, because NLopt requires that callback
signature even when gradients are not used.

The regression test is `tests/test_api_parameter_contracts.py`.

## Verified Candidate List

These candidates came from the initial static scan and are now resolved.

| Function | Parameter | Status | Required verification |
| --- | --- | --- | --- |
| `analyze_and_plot` | `include_calibrated_MACD` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `include_calibrated_RSI` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `indicators_signal_calcualtion` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `enabled_multiprocessing` | removed | Removed only from `analyze_and_plot`; still active in optimizer workflows. |
| `multiperiod_analysis` | `pars_from_opt_file` | removed | Removed after code and notebook scan confirmed no body usage. |
| `multiperiod_analysis` | `files_path_name` | removed | Removed after code and notebook scan confirmed no body usage. |
| `indict_MACD_SGMACD` | `signals_results` | removed | Function mutates and returns `data`; no caller needed a separate `signals_results` parameter. |
| `indict_RSI_SG_smooth_RSI` | `signals_results` | removed | Function mutates and returns `data`; no caller needed a separate `signals_results` parameter. |
| `indict_centered_average_deltas` | `signals_results` | removed | Function mutates and returns `data`; no caller needed a separate `signals_results` parameter. |
| `rebuilt_signal_zeros` | `debug` | removed | Removed after verifying only internal calls and running QQQ golden coverage. |
| `CDC_vs_detrended_correlation` | `data` | removed | Function builds a local one-row periods dataframe and calls `multiperiod_analysis`; `data` was not read. |
| `CDC_vs_detrended_correlation` | `lowess_k` | removed | Not routed to `multiperiod_analysis`; keeping it implied behavior that did not exist. |
| `CDC_vs_detrended_correlation` | `best_fit_start_back_period` | removed | Not routed to `multiperiod_analysis`; keeping it implied behavior that did not exist. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_fixed_periods` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_sg_filter_window` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `RSI_cycles_analysis_type` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `show_charts` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `source_type` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `data_column_name` | removed | Removed because the current workflow is hard-coded to `Close`. |
| `get_min_max_analysis_df` | `GoogleDriveMountPoint` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `index_column_name` | removed | Removed because resume loading parses `datetime` directly. |

## Parameters Intentionally Kept

| Function | Parameter | Status | Reason |
| --- | --- | --- | --- |
| `multiperiod_analysis` | `MultiAn_fitness_type_svg_smoothed` | active | Controls Savitzky-Golay smoothing of the reference detrended data before fitness evaluation. |
| `multiperiod_analysis` | `MultiAn_fitness_type_svg_filter` | active | Window length used by the smoothing path when `MultiAn_fitness_type_svg_smoothed=True`. |
| `multiperiod_analysis` | `enabled_multiprocessing` | mode-specific | Controls DEAP multiprocessing and C++ single/multicore selection. |
| `multiperiod_analysis` | `lowess_k` | mode-specific | Routed to `analyze_and_plot` only when LOWESS detrending is selected. |
| `multiperiod_analysis` | `linear_filter_window_size_multiplier` | mode-specific | Used to derive the detrending window for linear-style paths and optimization scenarios. |
| `multiperiod_analysis` | `discretization_steps` | mode-specific | Used by DEAP and native GA discretized initialization/mutation. |

## Removal Protocol

1. Find notebook and code references with `rg`.
2. Check whether the parameter is stored in `self` or passed under another name.
3. Add or update a golden test covering the relevant function.
4. Remove parameter only in a dedicated commit.
5. If public, keep a legacy wrapper or warning phase before final removal.

## Completed Removals

| Function | Parameter | Commit scope | Verification |
| --- | --- | --- | --- |
| nested `pick_extrema_near_target` inside `multiperiod_analysis` | `indices` | Internal helper only; call sites updated in the same block. | Rapid pytest suite and optional QQQ golden scenario. |
| `rebuilt_signal_zeros` | `debug` | Internal calls updated; the flag was never read. | Rapid pytest suite and optional QQQ golden scenario. |
| `analyze_and_plot` | `include_calibrated_MACD`, `include_calibrated_RSI`, `indicators_signal_calcualtion`, `enabled_multiprocessing` | Signature, internal calls, golden scenario and notebooks updated. | AST unused-parameter scan, py_compile, pytest. |
| `multiperiod_analysis` | `pars_from_opt_file`, `files_path_name`, `bb_delta_fixed_periods`, `bb_delta_sg_filter_window`, `RSI_cycles_analysis_type` | Signature, wrapper calls and notebooks updated; `enabled_multiprocessing` kept where active. | AST unused-parameter scan, py_compile, pytest. |
| `min_max_analysis_concatenated_dataframe` | `pars_from_opt_file`, `files_path_name`, `bb_delta_fixed_periods`, `bb_delta_sg_filter_window`, `RSI_cycles_analysis_type`, `show_charts` | Signature, callers and notebooks updated. | AST unused-parameter scan, py_compile, pytest. |
| `get_min_max_analysis_df` | `source_type`, `data_column_name`, `GoogleDriveMountPoint`, `index_column_name` | Signature and notebooks updated. | AST unused-parameter scan, py_compile, pytest. |
