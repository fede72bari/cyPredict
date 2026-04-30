import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_from_legacy_parent_library_path():
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
        cwd=REPO_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cyPredict" in result.stdout


def test_lowercase_import_facade_exposes_cypredict_alias():
    from cypredict import CyPredict, cyPredict

    assert CyPredict is cyPredict
    assert hasattr(CyPredict, "cyPredict")


def test_lowercase_import_facade_from_parent_path_matches_legacy_module():
    code = f"""
import sys
from pathlib import Path

repo = Path({str(REPO_ROOT)!r})
sys.path = [str(repo.parent)] + [
    path for path in sys.path
    if str(repo) not in path and str(repo.parent) not in path
]

from cyPredict import cyPredict as legacy_module
from cypredict import CyPredict

assert CyPredict is legacy_module.cyPredict
print(CyPredict)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cyPredict" in result.stdout or "cypredict" in result.stdout
