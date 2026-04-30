import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_optimization_smoke_matrix.py"
DEFAULT_SCENARIOS = [
    "monorange_qqq_hp",
    "monorange_es_hp",
    "multi_qqq_deap_amp",
    "multi_qqq_cpp_single_freq_phase",
    "multi_qqq_cpp_multicore_freq_phase",
]


def selected_scenarios():
    raw = os.environ.get("CYPREDICT_OPTIMIZATION_SCENARIOS")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return DEFAULT_SCENARIOS


@pytest.mark.skipif(
    os.environ.get("CYPREDICT_RUN_OPTIMIZATION_MATRIX") != "1",
    reason=(
        "Optimization smoke scenarios download live data and run slow optimizers; "
        "set CYPREDICT_RUN_OPTIMIZATION_MATRIX=1 to run."
    ),
)
@pytest.mark.parametrize("scenario", selected_scenarios())
def test_optimization_smoke_scenario(scenario):
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--scenario", scenario],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=360,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    json_line = next(
        line for line in reversed(result.stdout.splitlines())
        if line.startswith("{") and line.endswith("}")
    )
    report = json.loads(json_line)
    assert report["schema_version"] == 1
    assert len(report["scenarios"]) == 1

    summary = report["scenarios"][0]
    assert summary["scenario"] == scenario
    assert summary["elapsed_seconds"] > 0

    if summary["scenario_type"] == "monorange":
        assert summary["original_shape"][0] > 0
        assert summary["signals_shape"][0] > 0
        assert summary["signals_hash"]
    else:
        assert summary["dominant_shape"][0] > 0
        assert summary["composite_shape"][0] > 0
        assert summary["best_fitness"] == pytest.approx(float(summary["best_fitness"]))
        assert summary["dominant_hash"]
        assert summary["composite_hash"]
