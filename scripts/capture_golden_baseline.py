"""Capture a cyPredict golden baseline summary from a JSON scenario file.

The script is intentionally conservative: it does not edit project code and it
stores compact summaries/hashes of outputs rather than trying to serialize every
intermediate object in full.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_repo_root() -> None:
    """Make the package importable when this script is run from scripts/."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def add_repo_native_paths() -> None:
    """Make locally built native extension folders importable."""
    for relative in (
        "native/goertzel",
        "native/cyfitness",
        "native/cygaopt",
        "native/cygaopt_multicore",
        "native/genetic_optimization_legacy",
    ):
        candidate = REPO_ROOT / relative
        if candidate.exists():
            sys.path.insert(0, str(candidate))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def module_origin_and_hash(module_name: str) -> dict[str, Any]:
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return {"module": module_name, "origin": None, "sha256": None}
    origin = Path(spec.origin)
    return {
        "module": module_name,
        "origin": str(origin),
        "sha256": sha256_file(origin),
    }


def json_safe(value: Any) -> Any:
    """Convert common scientific Python scalar values into JSON-safe values."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if pd is not None and isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def summarize_dataframe(df: Any) -> dict[str, Any]:
    import pandas as pd

    hash_strategy = "pandas_hash"
    try:
        hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except TypeError:
        hash_strategy = "json_split"
        hashed = df.to_json(date_format="iso", orient="split", default_handler=str).encode("utf-8")
    return {
        "kind": "DataFrame",
        "shape": list(df.shape),
        "columns": [str(column) for column in df.columns],
        "index_name": json_safe(df.index.name),
        "index_dtype": str(df.index.dtype),
        "first_index": json_safe(df.index[0]) if len(df.index) else None,
        "last_index": json_safe(df.index[-1]) if len(df.index) else None,
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "hash": sha256_bytes(hashed),
        "hash_strategy": hash_strategy,
        "head": json.loads(df.head(5).to_json(date_format="iso", orient="split", default_handler=str)),
        "tail": json.loads(df.tail(5).to_json(date_format="iso", orient="split", default_handler=str)),
    }


def summarize_series(series: Any) -> dict[str, Any]:
    import pandas as pd

    hash_strategy = "pandas_hash"
    try:
        hashed = pd.util.hash_pandas_object(series, index=True).values.tobytes()
    except TypeError:
        hash_strategy = "json_split"
        hashed = series.to_json(date_format="iso", orient="split", default_handler=str).encode("utf-8")
    return {
        "kind": "Series",
        "name": json_safe(series.name),
        "shape": [len(series)],
        "dtype": str(series.dtype),
        "first_index": json_safe(series.index[0]) if len(series.index) else None,
        "last_index": json_safe(series.index[-1]) if len(series.index) else None,
        "hash": sha256_bytes(hashed),
        "hash_strategy": hash_strategy,
        "head": json.loads(series.head(5).to_json(date_format="iso", orient="split", default_handler=str)),
        "tail": json.loads(series.tail(5).to_json(date_format="iso", orient="split", default_handler=str)),
    }


def summarize_array(array: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(array)
    contiguous = np.ascontiguousarray(arr)
    return {
        "kind": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "hash": sha256_bytes(contiguous.tobytes()),
        "head": [json_safe(x) for x in arr.reshape(-1)[:5]],
        "tail": [json_safe(x) for x in arr.reshape(-1)[-5:]],
    }


def summarize(value: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return summarize_dataframe(value)
    if isinstance(value, pd.Series):
        return summarize_series(value)
    if isinstance(value, np.ndarray):
        return summarize_array(value)
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [summarize(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [summarize(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(key): summarize(item) for key, item in value.items()}}
    return {"kind": type(value).__name__, "value": json_safe(value)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture(config: dict[str, Any]) -> dict[str, Any]:
    add_repo_root()
    add_repo_native_paths()

    import cyPredict

    constructor_kwargs = config.get("constructor", {})
    method_name = config["method"]
    method_kwargs = config.get("method_kwargs", {})

    started = time.perf_counter()
    instance = cyPredict.cyPredict(**constructor_kwargs)
    result = getattr(instance, method_name)(**method_kwargs)
    elapsed_seconds = time.perf_counter() - started

    native_modules = [
        "goertzel",
        "cyfitness",
        "cyGAopt",
        "cyGAoptMultiCore",
        "talib",
        "nlopt",
    ]

    return {
        "schema_version": 1,
        "scenario_name": config["name"],
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "git_commit": current_git_commit(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "config_sha256": sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8")),
            "native_modules": [module_origin_and_hash(name) for name in native_modules],
        },
        "config": config,
        "result": summarize(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Scenario JSON file.")
    parser.add_argument("--output-dir", default=REPO_ROOT / "tests" / "golden" / "baselines", type=Path)
    parser.add_argument("--output-name", default=None, help="Optional output filename.")
    args = parser.parse_args()

    config = load_json(args.config)
    baseline = capture(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{config['name']}.json"
    output_path = args.output_dir / output_name
    output_path.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
