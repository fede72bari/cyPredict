"""Data loading helpers for :class:`cyPredict.cypredict.cyPredict`."""

import pandas as pd
import yfinance as yf


class DataMixin:
    """Load and normalize financial time series used by the legacy engine."""

    def download_finance_data(
        self,
        data_source,
        data_filename,
        ticker,
        data_start_date,
        data_end_date,
        data_timeframe
    ):
        """Load OHLCV data into ``self.data``.

        Parameters
        ----------
        data_source : {"yfinance", "file"}
            Source selector. ``"yfinance"`` downloads through Yahoo Finance.
            ``"file"`` reads a local CSV.
        data_filename : str or path-like, optional
            CSV path used only for ``data_source == "file"``. The file branch
            expects a column named exactly ``Datetime``; this column is parsed,
            set as index, and sorted.
        ticker : str
            Yahoo symbol used only when ``data_source == "yfinance"``.
        data_start_date, data_end_date : str or datetime-like
            Yahoo download bounds. ``data_end_date`` follows Yahoo semantics:
            it is normally exclusive for daily bars.
        data_timeframe : str
            Yahoo interval. Intraday values containing ``"m"`` or ``"h"``
            keep/localize timezone-aware indexes; daily/weekly/monthly values
            are normalized to timezone-naive indexes.

        Returns
        -------
        None
            The method mutates ``self.data``, ``self.data_start_date``,
            ``self.data_end_date`` and ``self.state``.

        Raises
        ------
        No exception is intentionally re-raised. Existing behavior catches
        exceptions, stores the message in ``self.state["data_state_msg"]`` and
        prints the error.

        Example
        -------
        >>> cp = cyPredict(print_activity_remarks=False)
        >>> cp.download_finance_data("file", r"C:\\data\\bars.csv", "QQQ", None, None, "1d")
        >>> cp.data.index.name
        'Datetime'
        """
        try:
            if data_source == 'yfinance':
                self.data = yf.download(ticker, start=data_start_date, end=data_end_date, interval=data_timeframe)
                self.data = self.data.xs(ticker, level=1, axis=1)

                # Intraday data keeps a timezone-aware index.
                if 'm' in data_timeframe or 'h' in data_timeframe:
                    if self.data.index.tz is None:
                        self.data.index = self.data.index.tz_localize(self.original_data_time_zone)
                # Daily and higher timeframes use timezone-naive indexes.
                elif 'd' in data_timeframe or 'w' in data_timeframe or 'M' in data_timeframe:
                    self.data.index = self.data.index.tz_localize(None)

            elif data_source == "file":
                self.data = pd.read_csv(data_filename)

                # Parse Datetime and use it as the sorted index.
                self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
                self.data.set_index("Datetime", inplace=True)
                self.data = self.data.sort_index()

                # Preserve existing notebook diagnostics.
                self.data_start_date = self.data.index.min()
                self.data_end_date = self.data.index.max()
                print("First df Datetime:", self.data_start_date)
                print("Last df Datetime:", self.data_end_date)

            else:
                self.state["data_state"] = 'error'
                self.state["data_state_msg"] = 'Not managed data source ' + data_source
                print('Error: not managed data source ' + data_source)

            if not self.data.empty:
                self.state["data_state"] = 'initialized'
                self.state["data_state_msg"] = 'Financial data is ready to be used.'

            else:
                self.state["data_state"] = 'error'
                self.state["data_state_msg"] = data_source + ' returned empty data, no error speficiation returned by the module. Look at standard output.'

        except Exception as e:
            self.state["data_state"] = 'error'
            self.state["data_state_msg"] = str(e)
            print(f"An error occurred in download_finance_data: {e}")
