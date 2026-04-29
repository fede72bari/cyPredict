# Known Baseline Blockers

These are issues observed while trying to create a tiny synthetic smoke scenario. They are documented here so they are not accidentally hidden by test scaffolding.

## `analyze_and_plot(detrend_type="none")`

Observed with `cyenv` on 2026-04-29:

- `detrend_type="none"` reaches a path where `detrended_data` is referenced before assignment.
- This appears to be an existing behavior issue, not introduced by the baseline tooling.

## `analyze_and_plot(other_correlations=False)`

Observed with `cyenv` on 2026-04-29:

- scoring expects columns such as `scaled_savgol_filter_delta_correlation` and `scaled_savgol_filter_delta_derivate_correlations`;
- those columns are not created when the correlation branch is skipped.

These should be handled after golden baselines are available, because changing them may alter behavior or expose previously masked paths.

