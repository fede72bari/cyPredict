import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_capture_golden_baseline_help_does_not_import_cypredict():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "capture_golden_baseline.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--config" in result.stdout
    assert "--output-dir" in result.stdout

