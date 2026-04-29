"""Technical-indicator helpers used by legacy cyPredict workflows."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import talib


class IndicatorsMixin:
    """Calculate MACD, RSI and centered-average helper columns."""

    def indict_MACD_SGMACD(self, data, data_column_name, dominant_period, macd_slow_ratio=26, macd_signal_ratio=9):
        """Add MACD and Savitzky-Golay MACD columns for a dominant period.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe containing ``data_column_name``.
        data_column_name : str
            Price/value column used as indicator input.
        dominant_period : float
            Dominant cycle period. The indicator period is half this value.
        macd_slow_ratio : float, default 26
            Ratio used to derive the slow MACD period from the fast period.
        macd_signal_ratio : float, default 9
            Ratio used to derive the MACD signal period from the fast period.

        Returns
        -------
        tuple
            ``(data, indicator_parameters)`` with new indicator columns added
            in place and metadata naming those columns.
        """
        indicators_period = int(dominant_period / 2)
        macd_fast = indicators_period
        macd_slow = int(macd_slow_ratio/12 * macd_fast)
        macd_signal = int(macd_signal_ratio/12 * macd_fast)

        data['macd_' + str(indicators_period)], data['macdSignal_' + str(indicators_period)], data['macdHist_' + str(indicators_period)] = talib.MACD(data[data_column_name], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)

        data['macdSignal_derivate_' + str(indicators_period)] = data['macdHist_' + str(indicators_period)].diff()
        data['macdSignal_derivate_' + str(indicators_period)] = data['macdSignal_derivate_' + str(indicators_period)] .fillna(0)

        savgol_filter_long_period = indicators_period*4
        savgol_filter_short_period = indicators_period*2

        data['savgol_filter_long_' + str(indicators_period)] = savgol_filter(data[data_column_name], int(savgol_filter_long_period), 2)
        data['savgol_filter_short_' + str(indicators_period)] = savgol_filter(data[data_column_name], int(savgol_filter_short_period), 2)
        data['savgol_MACD_' + str(indicators_period)] = data['savgol_filter_short_' + str(indicators_period)] - data['savgol_filter_long_' + str(indicators_period)]
        data['savgol_MACD_signal_' + str(indicators_period)] = savgol_filter(data['savgol_MACD_' + str(indicators_period)] , int(indicators_period*2), 2)
        data['savgol_MACD_hist_' + str(indicators_period)] = data['savgol_MACD_' + str(indicators_period)] - data['savgol_MACD_signal_' + str(indicators_period)]

        indicator_parameters = {
                                'dominant_period': dominant_period,
                                'indicators_period': indicators_period,
                                'MACD_savgol_filter_long_period': savgol_filter_long_period,
                                'MACD_savgol_filter_short_period': savgol_filter_short_period,
                                'macd_fast_period': macd_fast,
                                'macd_slow_period': macd_slow,
                                'macd_signal_period': macd_signal,
                                'macd_name': 'macd_' + str(indicators_period),
                                'macdSignal_name': 'macdSignal_' + str(indicators_period),
                                'macdHist_name': 'macdHist_' + str(indicators_period),
                                'macdSignal_derivate_name': 'macdSignal_derivate_' + str(indicators_period),
                                'savgol_MACD_name': 'savgol_MACD_' + str(indicators_period),
                                'savgol_MACD_signal_name': 'savgol_MACD_signal_' + str(indicators_period),
                                'savgol_MACD_hist_name': 'savgol_MACD_hist_' + str(indicators_period)
                                }

        return data, indicator_parameters

    def indict_RSI_SG_smooth_RSI(self, data, data_column_name, end_rebuilt_signal_index, dominant_period=10):
        """Add RSI and smoothed RSI derivative columns for a dominant period.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe containing ``data_column_name``.
        data_column_name : str
            Price/value column used as RSI input.
        end_rebuilt_signal_index : int
            Last index to fill with smoothed RSI values.
        dominant_period : float, default 10
            Dominant cycle period. The indicator period is half this value.

        Returns
        -------
        tuple
            ``(data, indicator_parameters)`` with new indicator columns added
            in place and metadata naming those columns.
        """
        data_len = len(data)
        indicators_period = int(dominant_period / 2)
        data['RSI_' + str(indicators_period)] = talib.RSI(data[data_column_name], indicators_period)
        data['RSI_' + str(indicators_period)] = data['RSI_' + str(indicators_period)] .fillna(0)

        polyorder = 2
        if indicators_period < 3:
            polyorder = 1

        data['smoothed_RSI_' + str(indicators_period)] = pd.Series([np.nan] * data_len)
        data.iloc[0:end_rebuilt_signal_index+1, data.columns.get_loc('smoothed_RSI_' + str(indicators_period))] = savgol_filter(data['RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1], indicators_period, polyorder)

        data['smoothed_RSI_derivate_' + str(indicators_period)] = pd.Series([np.nan] * data_len)
        data['smoothed_RSI_derivate_' + str(indicators_period)] = data['smoothed_RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1].diff()

        indicator_parameters = {
                                'dominant_period': dominant_period,
                                'indicator_period': indicators_period,
                                'RSI_name': 'RSI_' + str(indicators_period),
                                'smoothed_RSI_name': 'smoothed_RSI_' + str(indicators_period),
                                'smoothed_RSI_derivate_name': 'smoothed_RSI_derivate_' + str(indicators_period)
                               }

        return data, indicator_parameters

    def indict_centered_average_deltas(self, data, data_column_name, dominant_period=10):
        """Add centered-average delta columns for a dominant period.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe containing ``data_column_name``.
        data_column_name : str
            Price/value column used for rolling averages.
        dominant_period : float, default 10
            Dominant cycle period controlling long/short average windows.

        Returns
        -------
        tuple
            ``(data, indicator_parameters)`` with the delta column added in
            place and metadata naming that column.
        """
        long_average = data[data_column_name].rolling(window=round(dominant_period), center=True).mean()
        long_average.fillna(0, inplace=True)
        short_average = data[data_column_name].rolling(window=round(dominant_period/2), center=True).mean()
        short_average.fillna(0, inplace=True)

        averages_delta = long_average - short_average

        delta_column_name = 'centered_averages_delta_' + str(round(dominant_period))

        data[delta_column_name] = averages_delta

        indicator_parameters =  {
                                    'dominant_period': dominant_period,
                                    'long_average_period': round(dominant_period),
                                    'short_average_period': round(dominant_period/2),
                                    'centered_averages_delta_name': 'centered_averages_delta_' + str(round(dominant_period))
                                }

        return data, indicator_parameters
