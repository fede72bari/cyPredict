import json
import math
import os
from pathlib import Path

import pytest

from scripts.capture_golden_baseline import capture, load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = REPO_ROOT / "tests" / "golden" / "scenarios" / "qqq_eod_analyze_and_plot.json"
BASELINE_PATH = REPO_ROOT / "tests" / "golden" / "baselines" / "qqq_eod_analyze_and_plot.json"
FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-3


def assert_json_close(actual, expected, path="root"):
    if path.endswith(".hash"):
        return
    if isinstance(actual, float) and isinstance(expected, float):
        if math.isnan(actual) and math.isnan(expected):
            return
        assert math.isclose(actual, expected, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL), path
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            assert_json_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_json_close(actual_item, expected_item, f"{path}[{index}]")
        return
    assert actual == expected, path


@pytest.mark.skipif(
    os.environ.get("CYPREDICT_RUN_GOLDEN") != "1",
    reason="Golden cyPredict scenarios download data and import the full package; set CYPREDICT_RUN_GOLDEN=1 to run.",
)
def test_qqq_eod_analyze_and_plot_matches_golden_baseline():
    config = load_json(SCENARIO_PATH)
    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    actual = capture(config)

    assert actual["scenario_name"] == expected["scenario_name"]
    assert actual["metadata"]["config_sha256"] == expected["metadata"]["config_sha256"]
    assert_json_close(actual["result"], expected["result"])
