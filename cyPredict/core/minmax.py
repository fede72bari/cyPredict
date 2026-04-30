"""Min/max feature extraction workflows for projected cycle signals."""

import os

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, savgol_filter


class MinMaxMixin:
    """Build wide min/max analysis datasets used by backtests and notebooks."""

    def min_max_analysis(
        self,
        data,
        current_time_idx,
        suffix_col_name,
        N_elements=10,
        delta_comparison_serie=None,
        comparison_serie_name=''
    ):
        """Describe extrema around the current index for one signal.

        Parameters
        ----------
        data : pandas.Series
            Signal to inspect. The method finds local minima and maxima with
            first-neighbor extrema detection.
        current_time_idx : int
            Positional index representing the current analysis bar inside
            ``data``.
        suffix_col_name : str
            Prefix used for generated output columns. Past extrema receive
            ``-N`` labels and future/current extrema receive ``+N`` labels.
        N_elements : int, default 10
            Maximum number of extrema retained before and after
            ``current_time_idx``.
        delta_comparison_serie : pandas.Series or sequence, optional
            Optional reference series used to compute historical error columns.
            Only extrema at or before ``current_time_idx`` get an error value.
        comparison_serie_name : str, default ""
            Label inserted in error and contiguous-delta column names. It is
            meaningful only when ``delta_comparison_serie`` is provided.

        Returns
        -------
        pandas.DataFrame
            A one-row dataframe containing extrema distance, type, value,
            optional comparison metrics and a ``CDC_trend`` field.
        """
        cdc_min_max_indices = np.concatenate((
            argrelextrema(data.values, np.less, order=1)[0],
            argrelextrema(data.values, np.greater, order=1)[0]
        ))
        cdc_min_max_indices.sort()

        indices_before = cdc_min_max_indices[cdc_min_max_indices < current_time_idx][-N_elements:]
        indices_after = cdc_min_max_indices[cdc_min_max_indices >= current_time_idx][:N_elements]
        indices_before_after = np.concatenate((indices_before, indices_after))

        data_dict = {}
        previous_cdc_value = None

        for i, idx in enumerate(indices_before_after):
            abs_label_idx = abs(i - N_elements)
            col_prefix = (
                f'{suffix_col_name}+{abs_label_idx + 1}'
                if idx >= current_time_idx
                else f'{suffix_col_name}-{abs_label_idx}'
            )

            cdc_type = 1 if data[idx] > data[idx - 1] and data[idx] > data[idx + 1] else -1
            cdc_value = data[idx]

            data_dict[f'{col_prefix}_idx_delta'] = idx - current_time_idx
            data_dict[f'{col_prefix}_type'] = cdc_type
            data_dict[f'{col_prefix}_value'] = cdc_value

            if delta_comparison_serie is not None:
                if idx <= current_time_idx:
                    cdc_orig_delta = cdc_value - delta_comparison_serie[idx]
                    data_dict[f'{col_prefix}_{comparison_serie_name}_error'] = cdc_orig_delta

                if previous_cdc_value is not None:
                    data_dict[f'{col_prefix}_{comparison_serie_name}_contiguos_peaks_delta'] = abs(cdc_value - previous_cdc_value)

            previous_cdc_value = cdc_value

        cdc_type = data_dict[suffix_col_name + '-1_type'] if suffix_col_name + '-1_type' in data_dict else np.nan
        cdc_trend = 1 if cdc_type == -1 else -1

        data_dict[suffix_col_name + 'CDC_trend'] = cdc_trend
        data_df = pd.DataFrame([data_dict])

        return data_df

    def min_max_analysis_concatenated_dataframe(
        self,
        data_column_name,
        current_date,
        periods_pars,
        population_n=10,
        CXPB=0.7,
        MUTPB=0.3,
        NGEN=400,
        MultiAn_fitness_type="mse",
        MultiAn_fitness_type_svg_smoothed=False,
        MultiAn_fitness_type_svg_filter=4,
        reference_detrended_data="less_detrended",
        opt_algo_type='cpp_genetic_amp_freq_phase',
        amplitudes_inizialization_type="all_equal_middle_value",
        frequencies_ft=True,
        phases_ft=False,
        detrend_type='hp_filter',
        cut_to_date_before_detrending=True,
        linear_filter_window_size_multiplier=1.85,
        period_related_rebuild_range=True,
        period_related_rebuild_multiplier=2,
        discretization_steps=1000,
        lowess_k=6,
        windowing='kaiser',
        kaiser_beta=1,
        enabled_multiprocessing=True,
        N_elements_prices_CDC=6,
        N_elements_goertzel_CDC=3,
        N_elements_alignmentsKPI_CDC=10,
        N_elements_weigthed_alignmentsKPI_CDCC=10,
    ):
        """Build one min/max feature row for a single analysis date.

        This wrapper calls ``multiperiod_analysis`` for ``current_date`` and
        converts the resulting projected signals into a wide dataframe of
        local extrema features. It is used by ``get_min_max_analysis_df`` to
        create incremental CSV datasets for backtesting and model features.

        Parameters
        ----------
        data_column_name : str
            Price column analyzed by the underlying cycle workflow.
        current_date : str or datetime-like
            Anchor date/bar for the analysis.
        periods_pars : pandas.DataFrame
            Period-range table passed directly to ``multiperiod_analysis``.
        population_n, CXPB, MUTPB, NGEN : int or float
            Optimizer controls for the underlying multi-period refit.
        MultiAn_fitness_type, MultiAn_fitness_type_svg_smoothed,
        MultiAn_fitness_type_svg_filter, reference_detrended_data
            Fitness/reference controls passed through to ``multiperiod_analysis``.
        opt_algo_type, amplitudes_inizialization_type, frequencies_ft, phases_ft
            Optimizer branch and variable controls. Frequency/phase flags are
            meaningful only for algorithms that optimize those dimensions.
        detrend_type, cut_to_date_before_detrending, linear_filter_window_size_multiplier,
        period_related_rebuild_range, period_related_rebuild_multiplier,
        discretization_steps, lowess_k, windowing, kaiser_beta
            Detrending, rebuild-range and transform controls passed through to
            the underlying analysis.
        enabled_multiprocessing : bool, default True
            Enables multiprocessing where supported by the selected optimizer.
        N_elements_prices_CDC, N_elements_goertzel_CDC,
        N_elements_alignmentsKPI_CDC, N_elements_weigthed_alignmentsKPI_CDCC : int
            Number of past/future extrema features retained for each signal
            family.
        Returns
        -------
        pandas.DataFrame
            A one-row dataframe containing ``datetime``, ``best_fitness_value``,
            OHLCV values, derived price deltas, projected extrema descriptors
            and cycle-alignment trend fields.
        """
        datetime_df = pd.DataFrame()
        best_fitness_value_df = pd.DataFrame()

        max_datetime = self.data.index[self.data.index <= current_date].max()
        current_date_idx = self.data.index.get_loc(max_datetime)

        elaborated_data_df, signals_results_df, composite_signal, configurations, bb_delta, cdc_rsi, index_of_max_time_for_cd, scaled_signals, best_fitness_value = self.multiperiod_analysis(
            data_column_name=data_column_name,
            current_date=current_date,
            periods_pars=periods_pars,
            population_n=population_n,
            CXPB=CXPB,
            MUTPB=MUTPB,
            NGEN=NGEN,
            MultiAn_fitness_type=MultiAn_fitness_type,
            MultiAn_fitness_type_svg_smoothed=MultiAn_fitness_type_svg_smoothed,
            MultiAn_fitness_type_svg_filter=MultiAn_fitness_type_svg_filter,
            reference_detrended_data=reference_detrended_data,
            opt_algo_type=opt_algo_type,
            amplitudes_inizialization_type=amplitudes_inizialization_type,
            frequencies_ft=frequencies_ft,
            phases_ft=phases_ft,
            detrend_type=detrend_type,
            cut_to_date_before_detrending=cut_to_date_before_detrending,
            linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
            period_related_rebuild_range=period_related_rebuild_range,
            period_related_rebuild_multiplier=period_related_rebuild_multiplier,
            discretization_steps=discretization_steps,
            lowess_k=lowess_k,
            windowing=windowing,
            kaiser_beta=kaiser_beta,
            enabled_multiprocessing=enabled_multiprocessing,
            show_charts=False,
        )

        datetime_df['datetime'] = [current_date]
        best_fitness_value_df['best_fitness_value'] = [best_fitness_value]

        min_max_prices_CDC_analysis = self.min_max_analysis(
            data=scaled_signals['scaled_composite_signal'],
            delta_comparison_serie=scaled_signals['scaled_detrended'],
            comparison_serie_name='detrended_prices',
            current_time_idx=index_of_max_time_for_cd,
            suffix_col_name='prices_CDC_min_max_',
            N_elements=N_elements_prices_CDC
        )

        min_max_goertzel_CDC_analysis = self.min_max_analysis(
            data=pd.Series(savgol_filter(scaled_signals['scaled_goertzel_composite_signal'], 30, 2)),
            delta_comparison_serie=scaled_signals['scaled_detrended'],
            comparison_serie_name='detrended_prices',
            current_time_idx=index_of_max_time_for_cd,
            suffix_col_name='goertzel_CDC_min_max_',
            N_elements=N_elements_goertzel_CDC
        )

        min_max_alignmentsKPI_CDC_analysis = self.min_max_analysis(
            data=scaled_signals['scaled_alignmentsKPI'],
            current_time_idx=index_of_max_time_for_cd,
            suffix_col_name='scaled_alignmentsKPI_',
            N_elements=N_elements_alignmentsKPI_CDC
        )

        min_max_weigthed_alignmentsKPI_CDC_analysis = self.min_max_analysis(
            data=scaled_signals['scaled_weigthed_alignmentsKPI'],
            current_time_idx=index_of_max_time_for_cd,
            suffix_col_name='weigthed_alignmentsKPI_',
            N_elements=N_elements_weigthed_alignmentsKPI_CDCC
        )

        base_data = pd.DataFrame(
            [self.data.iloc[current_date_idx][['Open', 'Low', 'High', 'Close', 'Volume']].values],
            columns=['Open', 'Low', 'High', 'Close', 'Volume']
        )

        base_data['CO'] = base_data['Close'] - base_data['Open']
        base_data['HL'] = base_data['High'] - base_data['Low']
        base_data['CL'] = base_data['Close'] - base_data['Low']
        base_data['CH'] = base_data['Close'] - base_data['High']
        base_data['HO'] = base_data['High'] - base_data['Open']
        base_data['OL'] = base_data['Open'] - base_data['Low']

        base_data['HL_Volume_effort'] = base_data['HL'] / base_data['Volume']

        if current_date_idx > 0:
            base_data['Open_delta'] = self.data.iloc[current_date_idx]['Open'] - self.data.iloc[current_date_idx - 1]['Open']
            base_data['Close_delta'] = self.data.iloc[current_date_idx]['Close'] - self.data.iloc[current_date_idx - 1]['Close']
            base_data['High_delta'] = self.data.iloc[current_date_idx]['High'] - self.data.iloc[current_date_idx - 1]['High']
            base_data['Low_delta'] = self.data.iloc[current_date_idx]['Low'] - self.data.iloc[current_date_idx - 1]['Low']
            base_data['Volume_delta'] = self.data.iloc[current_date_idx]['Volume'] - self.data.iloc[current_date_idx - 1]['Volume']

            base_data['Close_Volume_effort_delta'] = base_data['Close_delta'] / base_data['Volume_delta']

        else:
            base_data['Open_delta'] = 0
            base_data['Close_delta'] = 0
            base_data['High_delta'] = 0
            base_data['Low_delta'] = 0
            base_data['Volume_delta'] = 0

            base_data['Close_Volume_effort_delta'] = 0

        base_data['scaled_detrended'] = scaled_signals['scaled_detrended']

        concatenated_dataframe = pd.concat([
            datetime_df,
            best_fitness_value_df,
            base_data,
            min_max_prices_CDC_analysis,
            min_max_goertzel_CDC_analysis,
            min_max_alignmentsKPI_CDC_analysis,
            min_max_weigthed_alignmentsKPI_CDC_analysis
        ], axis=1)

        return concatenated_dataframe

    def get_min_max_analysis_df(
        self,
        cycles_parameters,
        current_date,
        lookback_periods,
        retrieve_pars_from_file=False,
        optimized_pars_filepath=None,
        min_period=None,
        max_period=None,
        population_n=10,
        CXPB=0.7,
        MUTPB=0.3,
        NGEN=400,
        resume=False,
        file_path='/My Drive',
        file_name='/min_max_prices_analysis.csv',
        opt_algo_type='cpp_genetic_amp_freq_phase',
        amplitudes_inizialization_type="all_equal_middle_value",
        frequencies_ft=True,
        phases_ft=False,
        detrend_type='hp_filter',
        cut_to_date_before_detrending=True,
        linear_filter_window_size_multiplier=1.85,
        period_related_rebuild_range=True,
        period_related_rebuild_multiplier=2,
        discretization_steps=1000,
        lowess_k=6,
        windowing=None,
        kaiser_beta=1,
        enabled_multiprocessing=True,
    ):
        """Create or resume an incremental min/max analysis dataframe.

        The method iterates over missing dates in ``self.data`` up to
        ``current_date``, calls ``min_max_analysis_concatenated_dataframe`` for
        each date, appends the resulting rows, and persists the CSV to
        ``file_path + file_name``.

        Parameters
        ----------
        cycles_parameters : pandas.DataFrame
            Period-range table used for every date processed.
        current_date : str or datetime-like
            Last date to consider. The date must exist in ``self.data.index``.
        lookback_periods : int
            Number of bars before ``current_date`` considered for incremental
            processing.
        retrieve_pars_from_file, optimized_pars_filepath : optional
            Parameters for workflows that derive cycle parameters from an
            optimization file.
        min_period, max_period : int, optional
            Optional filter bounds for retrieved optimization parameters.
        population_n, CXPB, MUTPB, NGEN : int or float
            Optimizer controls passed through to per-date analysis.
        resume : bool, default False
            If true and the target CSV exists, load it and process only missing
            dates.
        file_path, file_name : str
            Output CSV location. They are concatenated into ``file_path_name``.
        opt_algo_type, amplitudes_inizialization_type, frequencies_ft, phases_ft,
        detrend_type, cut_to_date_before_detrending, linear_filter_window_size_multiplier,
        period_related_rebuild_range, period_related_rebuild_multiplier,
        discretization_steps, lowess_k, windowing, kaiser_beta,
        enabled_multiprocessing
            Passed through to ``min_max_analysis_concatenated_dataframe``.
        Returns
        -------
        pandas.DataFrame
            The updated min/max analysis dataframe, also written to CSV after
            each appended row.
        """
        file_path_name = file_path + file_name

        if resume and os.path.exists(file_path_name):
            min_max_CDC_analysis_df = pd.read_csv(file_path_name, parse_dates=['datetime'])

            self.log_info("Existing min/max CSV loaded", function="get_min_max_analysis_df", file_path=file_path_name)
        else:
            min_max_CDC_analysis_df = pd.DataFrame()
            self.log_info("Creating new min/max dataframe", function="get_min_max_analysis_df", file_path=file_path_name, resume=resume)

        if pd.to_datetime(current_date) not in self.data.index:
            self.log_warning("Current date not found in data", function="get_min_max_analysis_df", current_date=current_date)
            return min_max_CDC_analysis_df

        start_idx = self.data.index.get_loc(pd.to_datetime(current_date)) - lookback_periods
        if start_idx < 0:
            start_idx = 0
        filtered_data = self.data.iloc[start_idx:self.data.index.get_loc(pd.to_datetime(current_date)) + 1]

        if not min_max_CDC_analysis_df.empty:
            min_max_CDC_analysis_df['datetime'] = pd.to_datetime(min_max_CDC_analysis_df['datetime'], errors='coerce')
            missing_dates = filtered_data.index.difference(min_max_CDC_analysis_df['datetime'])
            filtered_data = filtered_data.loc[missing_dates]

        else:
            self.log_info("Min/max CSV is empty; every filtered date is new", function="get_min_max_analysis_df")

        for date in filtered_data.index:
            if date.tzinfo is None:
                date_str = date.replace(tzinfo=pd.Timestamp.utcnow().tz).isoformat()
            else:
                date_str = date.isoformat()

            self.log_info(
                "Running min_max_analysis_concatenated_dataframe",
                function="get_min_max_analysis_df",
                date=date_str,
            )

            if retrieve_pars_from_file and optimized_pars_filepath is not None:
                cycles_parameters = self.get_most_updated_optimization_pars(optimized_pars_filepath, date_str)

            if min_period is not None:
                cycles_parameters = cycles_parameters[cycles_parameters['min_period'] >= min_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)

            if max_period is not None:
                cycles_parameters = cycles_parameters[cycles_parameters['max_period'] <= max_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)

            analyzed_row = self.min_max_analysis_concatenated_dataframe(
                data_column_name='Close',
                current_date=date_str,
                periods_pars=cycles_parameters,
                population_n=population_n,
                CXPB=CXPB,
                MUTPB=MUTPB,
                NGEN=NGEN,
                MultiAn_fitness_type="mse",
                MultiAn_fitness_type_svg_smoothed=False,
                MultiAn_fitness_type_svg_filter=4,
                reference_detrended_data="less_detrended",
                opt_algo_type=opt_algo_type,
                detrend_type=detrend_type,
                linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
                period_related_rebuild_range=period_related_rebuild_range,
                period_related_rebuild_multiplier=period_related_rebuild_multiplier,
                cut_to_date_before_detrending=cut_to_date_before_detrending,
                frequencies_ft=frequencies_ft,
                phases_ft=phases_ft,
                amplitudes_inizialization_type=amplitudes_inizialization_type,
                discretization_steps=discretization_steps,
                lowess_k=lowess_k,
                windowing=windowing,
                kaiser_beta=kaiser_beta,
                enabled_multiprocessing=enabled_multiprocessing,
            )

            if 'datetime' in analyzed_row.columns:
                analyzed_row['datetime'] = pd.to_datetime(analyzed_row['datetime'], errors='coerce')

            min_max_CDC_analysis_df = pd.concat([min_max_CDC_analysis_df, analyzed_row])

            min_max_CDC_analysis_df = min_max_CDC_analysis_df.sort_values(by='datetime')
            min_max_CDC_analysis_df = min_max_CDC_analysis_df.drop_duplicates(subset=['datetime'], keep='first')

            min_max_CDC_analysis_df.to_csv(file_path_name, index=False)
            self.log_info("Min/max CSV updated", function="get_min_max_analysis_df", file_path=file_path_name)

        return min_max_CDC_analysis_df
