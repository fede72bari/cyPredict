import json
from pathlib import Path

import pytest


BASELINE_DIR = Path(__file__).resolve().parent / "golden" / "baselines"


def test_committed_golden_artifacts_have_expected_schema():
    baseline_files = sorted(BASELINE_DIR.glob("*.json"))
    if not baseline_files:
        pytest.skip("No golden baseline JSON artifacts have been captured yet.")

    for path in baseline_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["scenario_name"]
        assert "metadata" in data
        assert "config" in data
        assert "result" in data
        assert data["metadata"]["git_commit"]
        assert data["metadata"]["config_sha256"]

