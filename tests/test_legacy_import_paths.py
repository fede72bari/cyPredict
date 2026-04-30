import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_from_legacy_parent_library_path(tmp_path):
    code = f"""
import sys
from pathlib import Path

repo = Path({str(REPO_ROOT)!r})
sys.path = [str(repo.parent)] + [
    path for path in sys.path
    if str(repo) not in path and str(repo.parent) not in path
]

from cyPredict import cyPredict

assert hasattr(cyPredict, "cyPredict")
print(cyPredict.cyPredict)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cyPredict" in result.stdout
