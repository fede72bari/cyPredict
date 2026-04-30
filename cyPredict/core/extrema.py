"""Extrema and alignment helpers for projected cycle signals."""

import numpy as np
import pandas as pd


class ExtremaMixin:
    """Evaluate projected extrema and cycle-alignment KPIs."""

    def trade_predicted_dominant_cicles_peaks(self,
                                               current_date,
                                               num_samples=400,
                                               final_kept_n_dominant_circles=4,
                                               min_period=30,
                                               max_period=80,
                                               hp_filter_lambda=170000):
        """Estimate P/L between first projected max and min after current date.

        Parameters
        ----------
        current_date : datetime-like
            Analysis anchor date.
        num_samples : int, default 400
            Number of historical samples passed to ``analyze_and_plot``.
        final_kept_n_dominant_circles : int, default 4
            Number of dominant cycles kept by the analysis.
        min_period, max_period : int
            Period range searched by the dominant-cycle analysis.
        hp_filter_lambda : float, default 170000
            HP filter lambda passed to ``analyze_and_plot``.
        Returns
        -------
        float
            Difference between the close at the first projected maximum and
            the close at the first projected minimum. Returns ``0`` when the
            required extrema or data are unavailable.
        """
        _, _, elaborated_data, _, _ = self.analyze_and_plot(
                                                data_column_name='Close',
                                                num_samples=num_samples,
                                                start_date=None,
                                                current_date=current_date,
                                                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                dominant_cicles_sorting_type='global_score',
                                                limit_n_harmonics=None,
                                                min_period=min_period,
                                                max_period=max_period,
                                                detrend_type='hp_filter',
                                                bartel_scoring_threshold=0,
                                                hp_filter_lambda=hp_filter_lambda,
                                                jp_filter_p=3,
                                                jp_filter_h=100,
                                                bartel_peaks_filtering=True,
                                                centered_averages=True,
                                                other_correlations=True,
                                                show_charts=False,
                                                print_report=False
                                                )

        if elaborated_data is None:
            value = 0

        else:
            from scipy.signal import argrelmax, argrelmin

            maxes = argrelmax(elaborated_data['composite_dominant_circles_signal'].to_numpy(), order=10)[0]
            mins = argrelmin(elaborated_data['composite_dominant_circles_signal'].to_numpy(), order=10)[0]
            current_date = current_date.replace(tzinfo=None)
            current_date_index = elaborated_data.index.get_loc(current_date)

            maxes = maxes[maxes > current_date_index]
            if len(maxes) > 0:
                first_max = maxes[0]
            else:
                return 0

            mins = mins[mins > current_date_index]
            if len(mins) > 0:
                first_min = mins[0]
            else:
                return 0

            value = elaborated_data['Close'].iloc[first_max+1] - elaborated_data['Close'].iloc[first_min+1]

            if pd.isna(value) or value is None:
                return 0

            else:
                return value

    def CDC_vs_detrended_correlation(self,
                                     current_date,
                                     num_samples,
                                     final_kept_n_dominant_circles,
                                     min_period,
                                     max_period,
                                     hp_filter_lambda,
                                     opt_algo_type='genetic_omny_frequencies',
                                     detrend_type='hp_filter',
                                     windowing=None,
                                     kaiser_beta=3,
                                     linear_filter_window_size_multiplier=0.7,
                                     period_related_rebuild_range=True,
                                     period_related_rebuild_multiplier=2.5):
        """Run a small multiperiod analysis and return its best fitness value."""
        cycles_parameters = pd.DataFrame(columns=['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda'])
        cycles_parameters.loc[len(cycles_parameters)] = [num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda]

        (elaborated_data_df,
         signals_results_df,
         composite_signal,
         configurations,
         bb_delta, cdc_rsi,
          index_of_max_time_for_cd,
          scaled_signals,
          best_fitness_value) = self.multiperiod_analysis(
                                                        data_column_name='Close',
                                                        current_date=current_date,
                                                        periods_pars=cycles_parameters,
                                                        population_n=10,
                                                        CXPB=0.7,
                                                        MUTPB=0.3,
                                                        NGEN=30,
                                                        MultiAn_fitness_type="mse",
                                                        MultiAn_fitness_type_svg_smoothed=False,
                                                        MultiAn_fitness_type_svg_filter=4,
                                                        reference_detrended_data="less_detrended",
                                                        enable_cycles_alignment_analysis=False,
                                                        opt_algo_type=opt_algo_type,
                                                        detrend_type=detrend_type,
                                                        windowing=windowing,
                                                        kaiser_beta=kaiser_beta,
                                                        linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
                                                        period_related_rebuild_range=period_related_rebuild_range,
                                                        period_related_rebuild_multiplier=period_related_rebuild_multiplier,
                                                        show_charts=False,
                                                        enabled_multiprocessing=False
                                                        )

        if composite_signal.isna().any().any():
            self.log_warning("Composite signal contains NaN values", function="CDC_vs_detrended_correlation")

        return best_fitness_value

    def CDC_vs_detrended_correlation_sum(self,
                                          data,
                                          last_date,
                                          periods_number,
                                          num_samples,
                                          final_kept_n_dominant_circles,
                                          min_period,
                                          max_period,
                                          hp_filter_lambda,
                                          opt_algo_type,
                                          detrend_type,
                                          windowing=None,
                                          kaiser_beta=3,
                                          linear_filter_window_size_multiplier=0.7,
                                          period_related_rebuild_range=True,
                                          period_related_rebuild_multiplier=2.5):
        """Average ``CDC_vs_detrended_correlation`` over a lookback window."""
        last_date_index = data.index.get_loc(last_date)

        fitness = 0
        count = 0

        for index in range(periods_number):
            rel_pos = last_date_index - periods_number + index
            current_date = data.index[rel_pos]

            self.log_debug(
                "Calling CDC_vs_detrended_correlation in lookback loop",
                function="CDC_vs_detrended_correlation_sum",
                last_date=last_date,
                current_date=current_date,
                loop_index=index,
                periods_number=periods_number,
                num_samples=num_samples,
                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                hp_filter_lambda=hp_filter_lambda,
                period_related_rebuild_range=period_related_rebuild_range,
                period_related_rebuild_multiplier=period_related_rebuild_multiplier,
                linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
            )

            start = self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].min()
            end = self.MultiAn_dominant_cycles_df['end_rebuilt_signal_index'].max()

            rebuilt_segment = rebuilt_signal[start:end]
            reference_segment = self.detrended_data[start:end]

            self.log_debug(
                "Fitness segment NaN check",
                function="CDC_vs_detrended_correlation_sum",
                start=start,
                end=end,
                rebuilt_nan_count=rebuilt_segment.isna().sum(),
                reference_nan_count=reference_segment.isna().sum(),
            )

            if rebuilt_segment.isna().any() or reference_segment.isna().any():
                self.log_warning("NaN detected during fitness evaluation", function="CDC_vs_detrended_correlation_sum")

            temp_fitness = self.CDC_vs_detrended_correlation(
                                                            current_date=current_date,
                                                            num_samples=num_samples,
                                                            final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                            min_period=min_period,
                                                            max_period=max_period,
                                                            hp_filter_lambda=hp_filter_lambda,
                                                            windowing=windowing,
                                                            kaiser_beta=kaiser_beta,
                                                            opt_algo_type=opt_algo_type,
                                                            detrend_type=detrend_type,
                                                            linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
                                                            period_related_rebuild_range=period_related_rebuild_range,
                                                            period_related_rebuild_multiplier=period_related_rebuild_multiplier
                                                           )

            fitness += temp_fitness

            if temp_fitness > 0:
                count += 1

        if count > 0:
            fitness = fitness / count

        return fitness

    def trade_predicted_dominant_cicles_peaks_sum(self,
                                                  data,
                                                  last_date='2022-08-26',
                                                  periods_number=200,
                                                  num_samples=76,
                                                  final_kept_n_dominant_circles=2,
                                                  min_period=10,
                                                  max_period=18,
                                                  hp_filter_lambda=17):
        """Aggregate projected-extrema P/L over a lookback window."""
        value_sum = 0
        max_loss = 0
        max_cumulative_loss = 0
        temp_max_cumulative_loss = 0
        last_date_index = data.index.get_loc(last_date)
        profits_sum = 0
        profits_count = 0
        losses_sum = 0
        losses_count = 0

        for index in range(periods_number):
            rel_pos = last_date_index - periods_number + index
            current_date = data.index[rel_pos]

            PL = self.trade_predicted_dominant_cicles_peaks(
                                                      current_date=current_date,
                                                      num_samples=num_samples,
                                                      final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                      min_period=min_period,
                                                      max_period=max_period,
                                                      hp_filter_lambda=hp_filter_lambda
                                                    )

            if PL is not None and not pd.isna(PL):
                value_sum += PL

                if PL < max_loss:
                    max_loss = PL

                if PL < 0:
                    temp_max_cumulative_loss += PL
                    losses_sum += PL
                    losses_count += 1

                if PL > 0:
                    temp_max_cumulative_loss = 0
                    profits_sum += PL
                    profits_count += 1

            if temp_max_cumulative_loss < max_cumulative_loss:
                max_cumulative_loss = temp_max_cumulative_loss

        return value_sum, max_loss, max_cumulative_loss, profits_sum, profits_count, losses_sum, losses_count

    def MultiAn_cyclesAlignKPI(self, signals, start_position, weights=None, periods=None):
        """Compute alignment KPI between component-cycle extrema.

        Parameters
        ----------
        signals : pandas.DataFrame
            Component signal columns to inspect.
        start_position : int
            First position where KPI values are calculated. Earlier positions
            are prefilled with zero.
        weights : sequence, optional
            Optional per-signal weights.
        periods : sequence, optional
            Optional per-signal periods used to normalize weighted KPI values.

        Returns
        -------
        tuple
            ``(kpi_series, weighted_kpi_series)``.
        """
        from scipy.signal import argrelmax, argrelmin

        last_position = len(signals)

        kpi_series = pd.Series([0] * start_position, dtype=np.int64)
        weigthed_kpi_series = pd.Series([0] * start_position, dtype=np.int64)

        peaks_min_df = {}
        peaks_max_df = {}

        for column in signals.columns:
            peaks_min_df[column] = argrelmin(signals[column].values)[0]
            peaks_max_df[column] = argrelmax(signals[column].values)[0]

        if weights is not None and len(weights) > 0:
            weigths_sum = sum(weights)

        else:
            weigths_sum = 0

        for position in range(start_position, last_position):
            kpi = 0
            weigthed_kpi = 0

            weigths_index = 0

            for column in signals.columns:
                peaks_min = peaks_min_df[column]
                peaks_max = peaks_max_df[column]

                peaks_min_before = [peak for peak in peaks_min if peak < position]
                peaks_max_before = [peak for peak in peaks_max if peak < position]

                peaks_min_after = [peak for peak in peaks_min if peak > position]
                peaks_max_after = [peak for peak in peaks_max if peak > position]

                previous_peak_index_min = max(peaks_min_before) if peaks_min_before else np.nan
                previous_peak_index_max = max(peaks_max_before) if peaks_max_before else np.nan

                next_peak_index_min = min(peaks_min_after) if peaks_min_after else np.nan
                next_peak_index_max = min(peaks_max_after) if peaks_max_after else np.nan

                if not pd.isnull(previous_peak_index_max) and not pd.isnull(previous_peak_index_min):
                    if (position - previous_peak_index_max) < (position - previous_peak_index_min):
                        previous_peak_index = previous_peak_index_max
                        previous_peak_type = 'max'

                    else:
                        previous_peak_index = previous_peak_index_min
                        previous_peak_type = 'min'

                elif not pd.isnull(previous_peak_index_max):
                    previous_peak_index = previous_peak_index_max
                    revious_peak_type = 'max'

                elif not pd.isnull(previous_peak_index_min):
                    previous_peak_index = previous_peak_index_min
                    previous_peak_type = 'min'

                else:
                    previous_peak_index = np.nan
                    previous_peak_type = np.nan

                if not pd.isnull(next_peak_index_max) and not pd.isnull(next_peak_index_min):
                    next_peak_index = next_peak_index_max if (next_peak_index_max - position) < (next_peak_index_min - position) else next_peak_index_min

                elif not pd.isnull(next_peak_index_max):
                    next_peak_index = next_peak_index_max

                elif not pd.isnull(next_peak_index_min):
                    next_peak_index = next_peak_index_min

                else:
                    next_peak_index = np.nan

                if not pd.isnull(previous_peak_index) and not pd.isnull(next_peak_index):
                    path_len = next_peak_index - previous_peak_index

                    remain_len = next_peak_index - position

                    percentage = remain_len / path_len

                    if previous_peak_type == 'min':
                        kpi -= percentage

                    if previous_peak_type == 'max':
                        kpi += percentage

                    if weights is not None and len(weights) > 0:
                        if periods is not None:
                            if previous_peak_type == 'min':
                                weigthed_kpi -= percentage * weights[weigths_index] / periods[weigths_index]

                            if previous_peak_type == 'max':
                                weigthed_kpi += percentage * weights[weigths_index] / periods[weigths_index]

                        else:
                            if previous_peak_type == 'min':
                                weigthed_kpi -= percentage * weights[weigths_index] / weigths_sum

                            if previous_peak_type == 'max':
                                weigthed_kpi += percentage * weights[weigths_index] / weigths_sum

                weigths_index += 1

            kpi_series = pd.concat([kpi_series, pd.Series([kpi])], ignore_index=True)
            weigthed_kpi_series = pd.concat([weigthed_kpi_series, pd.Series([weigthed_kpi])], ignore_index=True)

        return kpi_series, weigthed_kpi_series
