"""Signal reconstruction helpers for the legacy cycle-analysis engine."""

import numpy as np
import pandas as pd


class ReconstructionMixin:
    """Build and align reconstructed cycle signals."""

    def cicles_composite_signals(self, max_length_series, amplitudes, MultiAn_dominant_cycles_df, df_indexes_list, composite_signal_column_name):
        """Build a composite signal from dominant cycle rows and amplitudes.

        Parameters
        ----------
        max_length_series : int
            Total number of samples in the target composite signal.
        amplitudes : sequence
            Amplitude value for each dominant-cycle row.
        MultiAn_dominant_cycles_df : pandas.DataFrame
            Rows containing at least ``peak_periods``, ``peak_frequencies``,
            ``peak_phases`` and ``start_rebuilt_signal_index``.
        df_indexes_list : sequence
            Index used for the returned dataframe.
        composite_signal_column_name : str
            Name of the aggregate composite-signal column.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the aggregate column and one column per
            component cycle.
        """
        composite_signal = pd.DataFrame(index=df_indexes_list)
        composite_signal[composite_signal_column_name] = 0.0

        for index, row in MultiAn_dominant_cycles_df.iterrows():

            temp_period = row['peak_periods']
            new_column_name = composite_signal_column_name + '_refact_dominant_circle_signal_period_' + str(temp_period)
            composite_signal[new_column_name] = 0.0

            remanant_length = np.int64(max_length_series - row['start_rebuilt_signal_index'])
            time = np.linspace(0, remanant_length, remanant_length, endpoint=False)

            composite_signal.iloc[int(row['start_rebuilt_signal_index']):, composite_signal.columns.get_loc(new_column_name)] = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
            composite_signal[composite_signal_column_name] += composite_signal[new_column_name]

        return composite_signal

    def rebuilt_signal_zeros(self, signal, start_rebuilt_signal_index, data):
        """Pad a rebuilt cycle signal so it aligns with the data index.

        Parameters
        ----------
        signal : array-like
            Rebuilt sinusoidal signal segment starting at
            ``start_rebuilt_signal_index``.
        start_rebuilt_signal_index : int
            Position in ``data`` where ``signal`` should begin. Zeros are
            prepended before this index.
        data : pandas.DataFrame or sequence-like
            Reference data whose length defines the current observed range.

        Returns
        -------
        tuple
            ``(signal, projection_periods_extensions)``. The returned signal is
            left-padded with zeros and trimmed/padded to the observed data
            length when possible. ``projection_periods_extensions`` is the
            number of periods by which the rebuilt signal extends beyond
            ``data``; callers use it to extend the datetime index.

        Notes
        -----
        This function does not calculate cycle values. It only aligns an
        already calculated signal with the dataframe length and projection
        horizon.
        """
        total_length = len(data)

        rebuilt_sig_left_zeros = np.zeros(start_rebuilt_signal_index)
        signal = np.concatenate((rebuilt_sig_left_zeros, signal), axis=0)

        len_after_left = len(signal)

        if total_length > len_after_left:
            rebuilt_sig_right_zeros = np.zeros(total_length - len_after_left)
            signal = np.concatenate((signal, rebuilt_sig_right_zeros), axis=0)
            signal = signal[0:(total_length)]

        projection_periods_extetions = 0
        if len(signal) > total_length:
            projection_periods_extetions = len(signal) - total_length

        return signal, projection_periods_extetions
