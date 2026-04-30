"""Run cyPredict optimization smoke scenarios.

The scenarios are intentionally small and live-data based. They are not golden
baselines; they verify that the main optimizer branches complete, return finite
fitness values, and produce structurally valid outputs for notebook-like QQQ
and ES=F configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_repo_paths() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@dataclass(frozen=True)
class AssetConfig:
    ticker: str
    data_start_date: str
    data_end_date: str
    current_date: str
    monorange: dict[str, Any]
    periods: list[dict[str, Any]]


ASSETS: dict[str, AssetConfig] = {
    "qqq": AssetConfig(
        ticker="QQQ",
        data_start_date="2022-01-01",
        data_end_date="2024-01-01",
        current_date="2023-12-29",
        monorange={
            "num_samples": 256,
            "min_period": 10,
            "max_period": 80,
            "hp_filter_lambda": 1600,
            "detrend_type": "hp_filter",
            "lowess_k": 3,
        },
        periods=[
            {
                "num_samples": 128,
                "final_kept_n_dominant_circles": 1,
                "min_period": 10,
                "max_period": 30,
                "hp_filter_lambda": 1600,
            },
            {
                "num_samples": 192,
                "final_kept_n_dominant_circles": 1,
                "min_period": 30,
                "max_period": 60,
                "hp_filter_lambda": 6400,
            },
        ],
    ),
    "es": AssetConfig(
        ticker="ES=F",
        data_start_date="2022-01-01",
        data_end_date="2024-10-16",
        current_date="2024-10-15",
        monorange={
            "num_samples": 219,
            "min_period": 9,
            "max_period": 64,
            "hp_filter_lambda": 800000,
            "detrend_type": "hp_filter",
            "lowess_k": 6,
        },
        periods=[
            {
                "num_samples": 160,
                "final_kept_n_dominant_circles": 1,
                "min_period": 6,
                "max_period": 16,
                "hp_filter_lambda": 65,
            },
            {
                "num_samples": 320,
                "final_kept_n_dominant_circles": 1,
                "min_period": 16,
                "max_period": 32,
                "hp_filter_lambda": 1039,
            },
        ],
    ),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataframe_hash(df: Any) -> str | None:
    import pandas as pd

    if df is None:
        return None
    try:
        raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except Exception:
        raw = df.to_json(date_format="iso", orient="split", default_handler=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_safe(value: Any) -> Any:
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return None
        return json_safe(value.iloc[0])
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def scalar_fitness(value: Any) -> float:
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        value = value.get("loss", value.get("fitness"))
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[0]
    if isinstance(value, np.generic):
        value = value.item()
    value = float(value)
    if not math.isfinite(value):
        raise AssertionError(f"non-finite fitness: {value!r}")
    return value


def make_instance(asset: AssetConfig):
    import cyPredict

    return cyPredict.cyPredict(
        data_source="yfinance",
        ticker=asset.ticker,
        data_start_date=asset.data_start_date,
        data_end_date=asset.data_end_date,
        data_timeframe="1d",
        time_tracking=False,
        output_clearing=False,
        print_activity_remarks=False,
    )


def run_monorange(asset_name: str, detrend_type: str) -> dict[str, Any]:
    asset = ASSETS[asset_name]
    cp = make_instance(asset)
    kwargs = dict(asset.monorange)
    kwargs["detrend_type"] = detrend_type
    if detrend_type == "lowess":
        kwargs["hp_filter_lambda"] = None

    current_date, index_of_max_time_for_cd, original_data, signals_results, configuration = cp.analyze_and_plot(
        data_column_name="Close",
        transform_precision=0.01,
        start_date=None,
        current_date=asset.current_date,
        final_kept_n_dominant_circles=2,
        dominant_cicles_sorting_type="global_score",
        limit_n_harmonics=None,
        detrend_window=0,
        bartel_peaks_filtering=True,
        bartel_scoring_threshold=0,
        jp_filter_p=3,
        jp_filter_h=100,
        cut_to_date_before_detrending=True,
        windowing=None,
        kaiser_beta=5,
        centered_averages=True,
        other_correlations=True,
        show_charts=False,
        print_report=False,
        debug=False,
        time_tracking=False,
        **kwargs,
    )

    assert original_data is not None and len(original_data) > 0
    assert signals_results is not None and len(signals_results) > 0

    return {
        "scenario_type": "monorange",
        "asset": asset_name,
        "ticker": asset.ticker,
        "detrend_type": detrend_type,
        "current_date": str(current_date),
        "index_of_max_time_for_cd": int(index_of_max_time_for_cd),
        "original_shape": list(original_data.shape),
        "signals_shape": list(signals_results.shape),
        "signals_hash": dataframe_hash(signals_results),
        "config_keys": sorted(str(k) for k in configuration.keys()),
    }


def run_multirange(
    asset_name: str,
    opt_algo_type: str,
    frequencies_ft: bool,
    phases_ft: bool,
    enabled_multiprocessing: bool,
) -> dict[str, Any]:
    import pandas as pd

    asset = ASSETS[asset_name]
    cp = make_instance(asset)
    periods = pd.DataFrame(asset.periods)

    result = cp.multiperiod_analysis(
        data_column_name="Close",
        current_date=asset.current_date,
        periods_pars=periods,
        show_charts=False,
        population_n=4,
        CXPB=0.5,
        MUTPB=0.2,
        NGEN=2,
        MultiAn_fitness_type="mse",
        MultiAn_fitness_type_svg_smoothed=False,
        MultiAn_fitness_type_svg_filter=5,
        reference_detrended_data="less_detrended",
        opt_algo_type=opt_algo_type,
        amplitudes_inizialization_type="all_equal_middle_value",
        frequencies_ft=frequencies_ft,
        phases_ft=phases_ft,
        detrend_type="hp_filter",
        cut_to_date_before_detrending=True,
        period_related_rebuild_range=False,
        period_related_rebuild_multiplier=2.5,
        discretization_steps=20,
        enabled_multiprocessing=enabled_multiprocessing,
        time_tracking=False,
        print_activity_remarks=False,
    )

    (
        elaborated_data_df,
        signals_results_df,
        composite_signal,
        configurations,
        _bb_delta,
        _cdc_rsi,
        index_of_max_time_for_cd,
        scaled_signals,
        best_fitness_value,
    ) = result

    best_fitness = scalar_fitness(best_fitness_value)
    dominant = cp.MultiAn_dominant_cycles_df.copy()
    assert len(dominant) > 0
    assert composite_signal is not None and len(composite_signal) > 0
    assert signals_results_df is not None and len(signals_results_df) > 0
    assert math.isfinite(best_fitness)

    best_columns = [str(column) for column in dominant.columns if str(column).startswith("best_")]
    return {
        "scenario_type": "multirange",
        "asset": asset_name,
        "ticker": asset.ticker,
        "opt_algo_type": opt_algo_type,
        "frequencies_ft": frequencies_ft,
        "phases_ft": phases_ft,
        "enabled_multiprocessing": enabled_multiprocessing,
        "index_of_max_time_for_cd": int(index_of_max_time_for_cd),
        "best_fitness": best_fitness,
        "dominant_shape": list(dominant.shape),
        "dominant_columns": [str(column) for column in dominant.columns],
        "best_columns": best_columns,
        "dominant_hash": dataframe_hash(dominant),
        "composite_shape": list(composite_signal.shape),
        "composite_hash": dataframe_hash(composite_signal),
        "elaborated_count": len(elaborated_data_df),
        "signals_count": len(signals_results_df),
        "scaled_signal_keys": sorted(str(k) for k in scaled_signals.keys()),
        "config_count": len(configurations),
    }


def scenario_definitions() -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    for asset in ("qqq", "es"):
        scenarios[f"monorange_{asset}_hp"] = {
            "kind": "monorange",
            "asset": asset,
            "detrend_type": "hp_filter",
        }
        scenarios[f"monorange_{asset}_lowess"] = {
            "kind": "monorange",
            "asset": asset,
            "detrend_type": "lowess",
        }

    flag_sets = {
        "amp": (False, False),
        "freq": (True, False),
        "phase": (False, True),
        "freq_phase": (True, True),
    }
    for asset in ("qqq", "es"):
        scenarios[f"multi_{asset}_mono_amp"] = {
            "kind": "multirange",
            "asset": asset,
            "opt_algo_type": "mono_frequency",
            "frequencies_ft": False,
            "phases_ft": False,
            "enabled_multiprocessing": False,
        }
        for suffix, (freq, phase) in flag_sets.items():
            scenarios[f"multi_{asset}_deap_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "genetic_omny_frequencies",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": False,
            }
            scenarios[f"multi_{asset}_cpp_single_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "cpp_genetic_amp_freq_phase",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": False,
            }
            scenarios[f"multi_{asset}_cpp_multicore_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "cpp_genetic_amp_freq_phase",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": True,
            }
            scenarios[f"multi_{asset}_nlopt_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "nlopt_amplitudes_freqs_phases",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": False,
            }
            scenarios[f"multi_{asset}_tpe_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "tpe",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": False,
            }
            scenarios[f"multi_{asset}_atpe_{suffix}"] = {
                "kind": "multirange",
                "asset": asset,
                "opt_algo_type": "atpe",
                "frequencies_ft": freq,
                "phases_ft": phase,
                "enabled_multiprocessing": False,
            }
    return scenarios


SCENARIOS = scenario_definitions()
QUICK_SCENARIOS = [
    "monorange_qqq_hp",
    "monorange_es_hp",
    "multi_qqq_mono_amp",
    "multi_qqq_deap_amp",
    "multi_qqq_deap_freq_phase",
    "multi_qqq_cpp_single_freq_phase",
    "multi_qqq_cpp_multicore_freq_phase",
    "multi_qqq_tpe_amp",
    "multi_qqq_atpe_freq_phase",
    "multi_es_deap_freq_phase",
    "multi_es_cpp_single_freq_phase",
]


def run_scenario(name: str) -> dict[str, Any]:
    definition = SCENARIOS[name]
    started = time.perf_counter()
    if definition["kind"] == "monorange":
        result = run_monorange(definition["asset"], definition["detrend_type"])
    else:
        result = run_multirange(
            asset_name=definition["asset"],
            opt_algo_type=definition["opt_algo_type"],
            frequencies_ft=definition["frequencies_ft"],
            phases_ft=definition["phases_ft"],
            enabled_multiprocessing=definition["enabled_multiprocessing"],
        )
    result["scenario"] = name
    result["elapsed_seconds"] = time.perf_counter() - started
    result["definition_hash"] = sha256_text(json.dumps(definition, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available scenario names.")
    parser.add_argument("--scenario", action="append", help="Scenario name to run. Can be repeated.")
    parser.add_argument("--quick", action="store_true", help="Run the curated quick matrix.")
    parser.add_argument("--all", action="store_true", help="Run every scenario.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    if args.list:
        for name in sorted(SCENARIOS):
            print(name)
        return 0

    add_repo_paths()

    if args.all:
        names = sorted(SCENARIOS)
    elif args.quick:
        names = QUICK_SCENARIOS
    elif args.scenario:
        names = args.scenario
    else:
        parser.error("choose --scenario, --quick, --all or --list")

    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {', '.join(unknown)}")

    results = [run_scenario(name) for name in names]
    report = {
        "schema_version": 1,
        "python": sys.version,
        "python_executable": sys.executable,
        "scenarios": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True, default=json_safe))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
