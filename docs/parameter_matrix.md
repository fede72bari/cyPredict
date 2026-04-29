# Parameter Matrix - Initial Draft

Baseline inspected: `873ad825a4e4edbe4962b13362d56bcb68da5408`

Status labels:

- `active`: directly used in calculation or control flow.
- `routing`: passed to another function or stored for later use.
- `mode-specific`: meaningful only for selected algorithms or options.
- `legacy`: retained for older callers or notebooks.
- `unused-candidate`: not observed in direct body usage during initial static scan; requires confirmation.
- `remove-approved`: may be removed only after notebook scan and golden tests.

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
| `analyze_and_plot` | `include_calibrated_MACD` | unused-candidate | Check older notebooks and intended indicator workflow. |
| `analyze_and_plot` | `include_calibrated_RSI` | unused-candidate | Check older notebooks and intended indicator workflow. |
| `analyze_and_plot` | `indicators_signal_calcualtion` | unused-candidate | Typo suggests legacy flag; check notebooks before removal. |
| `analyze_and_plot` | `enabled_multiprocessing` | unused-candidate | Confirm no intended branch was removed. |
| `multiperiod_analysis` | `pars_from_opt_file` | unused-candidate | Historical notebooks pass this; likely legacy/routing candidate. |
| `multiperiod_analysis` | `files_path_name` | unused-candidate | Historical notebooks pass this; likely legacy/routing candidate. |
| `indict_MACD_SGMACD` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `indict_RSI_SG_smooth_RSI` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `indict_centered_average_deltas` | `signals_results` | unused-candidate | Check whether function was meant to mutate passed dataframe. |
| `rebuilt_signal_zeros` | `debug` | unused-candidate | Safe candidate after test; may become logging flag. |
| `CDC_vs_detrended_correlation` | `data` | unused-candidate | High-risk because public-ish function may rely on `self.data`. |
| `CDC_vs_detrended_correlation` | `lowess_k` | unused-candidate | Check if missing routing is a bug before removing. |
| `CDC_vs_detrended_correlation` | `best_fit_start_back_period` | unused-candidate | Check whether should be passed to `multiperiod_analysis`. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_fixed_periods` | unused-candidate | Historical result columns may have depended on this. |
| `min_max_analysis_concatenated_dataframe` | `bb_delta_sg_filter_window` | unused-candidate | Historical result columns may have depended on this. |
| `min_max_analysis_concatenated_dataframe` | `RSI_cycles_analysis_type` | unused-candidate | Historical indicator workflow candidate. |
| `min_max_analysis_concatenated_dataframe` | `show_charts` | unused-candidate | Likely routing/legacy. |
| `get_min_max_analysis_df` | `source_type` | unused-candidate | Type currently references `Drive`; verify Google Drive path history. |
| `get_min_max_analysis_df` | `data_column_name` | unused-candidate | Current call appears hard-coded to `Close`; may be a bug. |
| `get_min_max_analysis_df` | `GoogleDriveMountPoint` | unused-candidate | Legacy Colab/Drive workflow. |
| `get_min_max_analysis_df` | `index_column_name` | unused-candidate | Current CSV load path may have changed. |

## Removal Protocol

1. Find notebook and code references with `rg`.
2. Check whether the parameter is stored in `self` or passed under another name.
3. Add or update a golden test covering the relevant function.
4. Remove parameter only in a dedicated commit.
5. If public, keep a legacy wrapper or warning phase before final removal.

