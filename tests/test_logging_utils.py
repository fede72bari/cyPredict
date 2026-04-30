import json
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cyPredict" / "logging_utils.py"
SPEC = importlib.util.spec_from_file_location("cypredict_logging_utils", MODULE_PATH)
assert SPEC is not None
logging_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = logging_utils
SPEC.loader.exec_module(logging_utils)
CyPredictLogger = logging_utils.CyPredictLogger


def test_logger_emits_labeled_event_without_console(capsys):
    logger = CyPredictLogger(
        ticker="ES=F",
        timeframe="5m",
        run_id="test123",
        log_to_console=False,
    )

    event = logger.emit(
        "started",
        category="processing",
        level="INFO",
        function="unit_test",
        rows=10,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert event is not None
    assert event.run_id == "test123"
    assert event.ticker == "ES=F"
    assert event.timeframe == "5m"
    assert event.category == "processing"
    assert event.function == "unit_test"
    assert event.context["rows"] == 10


def test_logger_writes_log_and_jsonl():
    log_dir = Path(__file__).resolve().parents[1] / "runtime_test_outputs" / "logging"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    logger = CyPredictLogger(
        ticker="NQ=F",
        timeframe="15m",
        run_id="abc",
        log_dir=log_dir,
        log_to_console=False,
        log_to_file=True,
    )

    try:
        logger.timing("checkpoint", function="analysis", step=1)

        assert logger.log_path is not None
        assert logger.jsonl_path is not None
        assert logger.log_path.exists()
        assert logger.jsonl_path.exists()
        assert "checkpoint" in logger.log_path.read_text(encoding="utf-8")

        row = json.loads(logger.jsonl_path.read_text(encoding="utf-8").strip())
        assert row["run_id"] == "abc"
        assert row["category"] == "timing"
        assert row["function"] == "analysis"
        assert row["context"]["step"] == 1
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def test_logger_filters_below_min_level():
    logger = CyPredictLogger(min_level="WARNING", log_to_console=False)

    assert logger.info("hidden") is None
    assert logger.warning("visible") is not None


def test_logger_rejects_unknown_category():
    logger = CyPredictLogger(log_to_console=False)

    with pytest.raises(ValueError):
        logger.emit("bad", category="unknown")
