# Parameter Matrix - Initial Draft

Baseline inspected: `873ad825a4e4edbe4962b13362d56bcb68da5408`

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

## Initial Unused-Candidate List

These candidates came from a static scan of function bodies and must be verified before changes.

| Function | Parameter | Status | Required verification |
| --- | --- | --- | --- |
| `analyze_and_plot` | `include_calibrated_MACD` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `include_calibrated_RSI` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `indicators_signal_calcualtion` | removed | Removed after code and notebook scan confirmed no body usage. |
| `analyze_and_plot` | `enabled_multiprocessing` | removed | Removed only from `analyze_and_plot`; still active in optimizer workflows. |
| `multiperiod_analysis` | `pars_from_opt_file` | removed | Removed after code and notebook scan confirmed no body usage. |
| `multiperiod_analysis` | `files_path_name` | removed | Removed after code and notebook scan confirmed no body usage. |
| `indict_MACD_SGMACD` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `indict_RSI_SG_smooth_RSI` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `indict_centered_average_deltas` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `rebuilt_signal_zeros` | `debug` | removed | Removed after verifying only internal calls and running QQQ golden coverage. |
| `CDC_vs_detrended_correlation` | `data` | unused-candidate | High-risk because public-ish function may rely on `self.data`. |
| `CDC_vs_detrended_correlation` | `lowess_k` | unused-candidate | Check if missing routing is a bug before removing. |
| `CDC_vs_detrended_correlation` | `best_fit_start_back_period` | unused-candidate | Check whether should be passed to `multiperiod_analysis`. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_fixed_periods` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_sg_filter_window` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `RSI_cycles_analysis_type` | removed | Removed after code and notebook scan confirmed no body usage. |
| `min_max_analysis_concatenated_dataframe` | `show_charts` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `source_type` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `data_column_name` | removed | Removed because the current workflow is hard-coded to `Close`. |
| `get_min_max_analysis_df` | `GoogleDriveMountPoint` | removed | Removed after code and notebook scan confirmed no body usage. |
| `get_min_max_analysis_df` | `index_column_name` | removed | Removed because resume loading parses `datetime` directly. |

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
