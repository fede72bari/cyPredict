import inspect

from cyPredict.core.multiperiod import MultiperiodMixin
from cyPredict.core.state import StateMixin
from cyPredict.logging_utils import CyPredictLogger


def test_multiperiod_analysis_accepts_logging_overrides():
    parameters = inspect.signature(MultiperiodMixin.multiperiod_analysis).parameters

    for name in ("log_level", "log_to_console", "log_to_file", "log_dir", "log_run_id"):
        assert name in parameters


def test_state_configure_logging_updates_existing_instance():
    state = object.__new__(StateMixin)
    state.state = {"ticker": "ES=F", "data_timeframe": "1d"}
    state.logger = CyPredictLogger(
        ticker="QQQ",
        timeframe="5m",
        min_level="WARNING",
        log_to_console=False,
        log_to_file=False,
    )

    logger = state.configure_logging(
        log_level="DEBUG",
        log_to_console=True,
        log_to_file=False,
        log_run_id="notebook-run",
    )

    assert logger.run_id == "notebook-run"
    assert logger.ticker == "ES=F"
    assert logger.timeframe == "1d"
    assert logger.min_level == "DEBUG"
    assert logger.log_to_console is True
    assert logger.log_to_file is False
