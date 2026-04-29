# Golden Baseline Scenarios

This document defines the first baseline tests needed before calculation cleanup.

## Scenario 1 - Daily Yahoo Data

- Data source: `yfinance`
- Candidate ticker: `SPY`
- Timeframe: `1d`
- Purpose: verify daily date handling, detrending, Goertzel period selection, and single/multirange output dimensions.
- Primary functions:
  - `cyPredict(...)`
  - `analyze_and_plot(...)`
  - `multiperiod_analysis(...)`

## Scenario 2 - Intraday Yahoo Data

- Data source: `yfinance`
- Candidate ticker: `ES=F` or `NQ=F`
- Timeframe: intraday, matching current notebook usage.
- Purpose: verify timezone-aware current dates, projection index extension, and min/max extraction.
- Primary functions:
  - `datetime_dateset_extend(...)`
  - `multiperiod_analysis(...)`
  - `min_max_analysis_concatenated_dataframe(...)`

## Scenario 3 - File-Based Intraday Data

- Data source: `file`
- Required columns: `Datetime`, `Open`, `High`, `Low`, `Close`, `Volume`
- Purpose: verify that CSV loading, date sorting, and index localization behave the same after cleanup.

## Scenario 4 - Native Optimization Path

- Native modules:
  - `goertzel`
  - `cyfitness`
  - `cyGAopt`
  - `cyGAoptMultiCore`
- Purpose: verify C/C++ bridge behavior for amplitude/frequency/phase optimization.
- Required controls:
  - seed where available;
  - module version/hash;
  - Python version and ABI.

## Values To Store

- Input parameters as JSON.
- Python version.
- Git commit.
- Native module filenames and hashes.
- Output dataframe schemas.
- Selected dominant periods.
- Best amplitudes/frequencies/phases.
- `best_fitness_value`.
- `index_of_max_time_for_cd`.
- `composite_signal` tail/head and full hash.
- `scaled_signals` schema and selected values.

## Comparison Rules

- Exact match: schemas, column order, index length, dates, categorical/string values.
- Numeric match: `numpy.testing.assert_allclose` with explicit tolerance.
- Genetic path: use deterministic seed where possible; otherwise compare stable invariants and acceptable ranges until native determinism is enforced.

## Tooling Added

Use `scripts/capture_golden_baseline.py` with the `cyenv` interpreter to capture compact baseline summaries. Scenario JSON templates and storage notes live under `tests/golden/`.
