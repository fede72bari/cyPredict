"""State and lifecycle helpers for :class:`cyPredict.cypredict.cyPredict`."""

from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..logging_utils import CyPredictLogger


class StateMixin:
    """Initialize shared state for the legacy cycle-analysis engine."""

    class Drive(Enum):
        local = 1
        GoogleDrive = 2

    class financialDataSource(Enum):
        csv_file = 1
        yfinance = 2

    def __init__(self,
                 data_source="yfinance",
                 data_filename=None,
                 ticker="SPY",
                 data_start_date="2004-01-01",
                 data_end_date=None,
                 data_timeframe="1d",
                 data_storage_path="\\cyPredict\\",
                 output_clearing=False,
                 log_level="WARNING",
                 log_to_console=False,
                 log_to_file=False,
                 log_dir="logs",
                 log_run_id=None):
        """Create an analysis instance and immediately initialize market data.

        Parameters
        ----------
        data_source : str, default "yfinance"
            Input provider. Supported values in the current implementation are
            ``"yfinance"`` and ``"file"``. ``"yfinance"`` calls
            ``yf.download`` with ``ticker``, ``data_start_date``,
            ``data_end_date`` and ``data_timeframe``. ``"file"`` reads
            ``data_filename`` as a CSV and expects a ``Datetime`` column.
        data_filename : str or path-like, optional
            CSV path used only when ``data_source == "file"``. The file must
            contain at least ``Datetime`` plus the price columns later selected
            by analysis methods, normally ``Open``, ``High``, ``Low``,
            ``Close`` and ``Volume``.
        ticker : str, default "SPY"
            Symbol passed to Yahoo Finance when ``data_source == "yfinance"``.
            It is also stored in ``self.state`` for logging and downstream
            reporting.
        data_start_date, data_end_date : str or datetime-like, optional
            Download bounds for Yahoo Finance. For deterministic baselines,
            prefer closed historical ranges such as ``2022-01-01`` to
            ``2024-01-01``. ``data_end_date=None`` requests data up to the
            provider default/current availability.
        data_timeframe : str, default "1d"
            Yahoo interval, for example ``"1d"``, ``"5m"`` or ``"1h"``. The
            value affects timezone handling in ``download_finance_data``.
        data_storage_path : str, default "\\cyPredict\\"
            Base path used by later file-writing/reporting helpers. It does not
            change the initial data download.
        output_clearing : bool, default False
            Legacy notebook flag retained for workflows that clear notebook
            output between long processing steps.
        log_level : {"DEBUG", "INFO", "WARNING", "ERROR"}, default "WARNING"
            Minimum structured logging level. ``INFO`` includes processing and
            timing events; ``DEBUG`` includes diagnostic events and intermediate
            objects.
        log_to_console : bool, default False
            Emit structured log lines to console.
        log_to_file : bool, default False
            Persist log lines and JSONL events under ``log_dir``.
        log_dir : str, default "logs"
            Directory used when ``log_to_file`` is true.
        log_run_id : str, optional
            Stable run identifier. When omitted, a short random id is created.

        Notes
        -----
        Construction has side effects: it downloads or reads data immediately
        and populates ``self.data`` and ``self.state``. For tests or worker
        jobs, instantiate once per independent data request.

        Example
        -------
        >>> cp = cyPredict(
        ...     data_source="yfinance",
        ...     ticker="QQQ",
        ...     data_start_date="2022-01-01",
        ...     data_end_date="2024-01-01",
        ...     data_timeframe="1d",
        ...     log_level="WARNING",
        ... )
        >>> cp.state["data_state"]
        'initialized'
        """

        self.data_source = data_source
        self.data_filename = data_filename
        self.ticker = ticker
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.data = []
        self.data_storage_path = data_storage_path
        self.logger = CyPredictLogger(
            ticker=ticker,
            timeframe=data_timeframe,
            log_dir=log_dir,
            run_id=log_run_id,
            min_level=log_level,
            log_to_console=log_to_console,
            log_to_file=log_to_file,
        )

        self.genOpt_last_date = ''
        self.genOpt_logarithmic_sequence = []
        self.genOpt_num_samples_min = 0
        self.genOpt_num_samples_max = 0
        self.genOpt_final_kept_n_dominant_circles_min = 0
        self.genOpt_final_kept_n_dominant_circles_max = 0
        self.genOpt_min_period_min = 0
        self.genOpt_min_period_max = 0
        self.genOpt_max_period_min = 0
        self.genOpt_max_period_max = 0
        self.genOpt_logarithmic_sequence = 0
        self.genOpt_periods_number = 0

        self.MultiAn_detrended_max = np.int64(0)
        self.MultiAn_detrended_min = np.int64(0)
        self.MultiAn_dominant_cycles_df = pd.DataFrame({
            'peak_frequencies': pd.Series(dtype='float64'),
            'peak_periods': pd.Series(dtype='float64'),
            'peak_phases': pd.Series(dtype='float64'),
            'start_rebuilt_signal_index': pd.Series(dtype='int64'),
            'end_rebuilt_signal_index': pd.Series(dtype='int64')
        })
        self.MultiAn_reference_detrended_data = []
        self.MultiAn_fitness_type = "all"

        self.state = {
            "data_source": data_source,
            "data_filename": data_filename,
            "ticker": ticker,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "data_timeframe": data_timeframe,
            "data_state": 'not initialized',
            "data_state_msg": '',
        }

        self.download_finance_data(
            self.state["data_source"],
            self.state["data_filename"],
            self.state["ticker"],
            self.state["data_start_date"],
            self.state["data_end_date"],
            self.state["data_timeframe"]
        )

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.output_clearing = output_clearing

    def is_log_enabled(self, level):
        return self.logger.is_enabled(level)

    def log_debug(self, message, *, function=None, **context):
        return self.logger.debug(message, function=function, **context)

    def log_info(self, message, *, function=None, **context):
        return self.logger.info(message, function=function, **context)

    def log_warning(self, message, *, function=None, **context):
        return self.logger.warning(message, function=function, **context)

    def log_error(self, message, *, function=None, **context):
        return self.logger.error(message, function=function, **context)

    def log_timing(self, message, *, function=None, **context):
        return self.logger.timing(message, function=function, **context)
