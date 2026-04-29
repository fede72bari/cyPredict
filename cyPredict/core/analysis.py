"""Single-range dominant-cycle analysis workflow."""

import math
import sys
import traceback

from goertzel import goertzel_DFT, goertzel_general_shortened
from IPython.display import display
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from scipy.signal import argrelextrema, argrelmax, argrelmin, find_peaks, savgol_filter
from scipy.signal.windows import kaiser, tukey
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.tsatools import detrend


class AnalysisMixin:
    """Run the single-range Goertzel/detrending cycle analysis."""

    def analyze_and_plot(self,
                         data = None,
                         data_column_name = 'Close',
                         transform_precision = 0.01,
                         num_samples = None,
                         start_date = None,
                         current_date = None,
                         final_kept_n_dominant_circles = 1,
                         dominant_cicles_sorting_type = 'global_score',
                         limit_n_harmonics = None,
                         min_period = 20,
                         max_period = 100,
                         detrend_type = 'none',
                         detrend_window = 0,
                         bartel_peaks_filtering = True,
                         bartel_scoring_threshold = 0.5,
                         hp_filter_lambda = 100,
                         jp_filter_p = 4,
                         jp_filter_h = 8,
                         cut_to_date_before_detrending = True,
                         lowess_k = 3,
                         windowing = None,
                         kaiser_beta = 5,
                         centered_averages = True,
                         other_correlations = False,
                         show_charts = False,
                         print_report = True,
                         debug = False,
                         time_tracking = False                         
                         ):
        """Run a single dominant-cycle analysis on one period range.

        This is the core single-range workflow used directly by notebooks and
        indirectly by ``multiperiod_analysis``. It prepares the selected price
        series, applies the requested detrending path, runs Goertzel analysis
        over ``min_period``..``max_period``, selects dominant cycles, rebuilds
        projected sinusoidal signals, and optionally computes correlation
        scores and plotting artifacts.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Input OHLCV/frame override. When omitted, ``self.data`` is used.
            The index is converted to ``DatetimeIndex`` and sorted in place on
            the working object.
        data_column_name : str, default "Close"
            Column analyzed for dominant cycles. Most notebook workflows use
            ``"Close"``.
        transform_precision : float, default 0.01
            Frequency-grid precision for Goertzel search. Smaller values
            increase resolution and runtime.
        num_samples, start_date, current_date : optional
            Window definition. Existing behavior expects exactly two of these
            three values to be meaningful. The common notebook path passes
            ``num_samples`` and ``current_date`` and leaves ``start_date`` as
            ``None``.
        final_kept_n_dominant_circles : int, default 1
            Number of selected dominant cycles kept after scoring/filtering.
        dominant_cicles_sorting_type : str, default "global_score"
            Ranking mode for dominant periods. ``"global_score"`` is the
            current notebook default.
        limit_n_harmonics : int, optional
            Optional cap on harmonics considered during peak processing.
        min_period, max_period : int
            Period search range. These bounds define the candidate cycle
            lengths and interact with ``num_samples``: very long periods need
            enough samples to be meaningful.
        detrend_type : {"none", "hp_filter", "linear", "lowess"}, default "none"
            Detrending mode. ``hp_filter_lambda`` is meaningful for
            ``"hp_filter"``. ``lowess_k`` is meaningful for ``"lowess"``.
            ``detrend_window`` is meaningful for linear detrending.
        detrend_window : int, default 0
            Window for linear detrending paths.
        bartel_peaks_filtering : bool, default True
            Enables Bartels-score filtering of peaks.
        bartel_scoring_threshold : float, default 0.5
            Minimum Bartels-related score used when filtering is enabled.
        hp_filter_lambda : float, default 100
            Lambda for the HP filter path.
        jp_filter_p, jp_filter_h : int
            Parameters for the Jurik-like filter branch used in existing
            extrema/correlation processing.
        cut_to_date_before_detrending : bool, default True
            When true, the analysis cuts data at ``current_date`` before
            detrending to avoid using future data in the detrend reference.
        lowess_k : int, default 3
            LOWESS factor, meaningful only with ``detrend_type == "lowess"``.
        windowing : {None, "tukey", "kaiser"}, optional
            Optional window function before transform. ``kaiser_beta`` matters
            only when ``windowing == "kaiser"``.
        kaiser_beta : float, default 5
            Kaiser window beta.
        centered_averages : bool, default True
            Enables centered-average derived columns used by correlations and
            scoring.
        other_correlations : bool, default False
            Enables additional correlation columns. Some legacy paths expect
            these columns to exist, so disabling this can expose missing-column
            behavior documented in ``docs/known_baseline_blockers.md``.
        show_charts : bool, default False
            Displays Plotly charts. It should not affect numerical outputs.
        print_report : bool, default True
            Prints tabular/details report.
        debug : bool, default False
            Enables selected debug prints.
        time_tracking : bool, default False
            Enables elapsed-time prints for this call.

        Returns
        -------
        tuple
            ``(current_date, index_of_max_time_for_cd, original_data,
            signals_results, configuration)``.

            ``original_data`` is the working dataframe with appended cycle and
            detrended columns. ``signals_results`` stores selected dominant
            cycle metadata, including period, frequency, amplitude, phase and
            rebuild indexes. ``configuration`` mirrors the main inputs used for
            the run.

        Example
        -------
        >>> cp = cyPredict(
        ...     data_source="yfinance", ticker="QQQ",
        ...     data_start_date="2022-01-01", data_end_date="2024-01-01",
        ...     data_timeframe="1d", print_activity_remarks=False)
        >>> current_date, idx, data, signals, config = cp.analyze_and_plot(
        ...     data_column_name="Close",
        ...     num_samples=256,
        ...     current_date="2023-12-29",
        ...     final_kept_n_dominant_circles=4,
        ...     min_period=10,
        ...     max_period=80,
        ...     detrend_type="hp_filter",
        ...     other_correlations=True,
        ...     show_charts=False,
        ...     print_report=False)
        """

        # Configuration
        configuration = {
          "data_column_name": data_column_name,
          "transform_precision": transform_precision,
          "num_samples": num_samples,
          "start_date": start_date,
          "current_date": current_date,
          "final_kept_n_dominant_circles": final_kept_n_dominant_circles,
          "limit_n_harmonics": limit_n_harmonics,
          "min_period": min_period,
          "max_period": max_period,
          "detrend_type": detrend_type,
          "detrend_window": detrend_window,
          "bartel_peaks_filtering": bartel_peaks_filtering,
          "bartel_scoring_threshold": bartel_scoring_threshold,
          "hp_filter_lambda": hp_filter_lambda,
          "jp_filter_p": jp_filter_p,
          "jp_filter_h": jp_filter_h,
          "lowess_k": lowess_k,
          "cut_to_date_before_detrending": cut_to_date_before_detrending
        }

        
        if(time_tracking):
            
            self.set_start_time()
            self.track_time('\nTime tracking started.')

        

        signals_results = pd.DataFrame()

        if(data is not None):
            if(data.empty): # or self.state["data_state"] != 'initialized'):
                print("Financial data not available.")
                return None, None, None, None, None

            else:
                original_data = data

        else:

            if(self.data.empty): # or self.state["data_state"] != 'initialized'):
                print("Financial data not available.")
                return None, None, None, None, None

            else:
                original_data = self.data


        # Check parameters constraints
        if((num_samples == None) &
           (start_date == None) &
           (current_date == None)):

            print("At least two of num_samples, start_date and current_date shall be not empty and valid.")

        if((num_samples != None) &
           (start_date != None) &
           (current_date != None)):

            print("Ambigouos number of not null paramenters: one of num_samples, start_date and current_date shall be empty.")

        if((final_kept_n_dominant_circles == None) & ((min_period == None) | (max_period == None))):
            print("final_kept_n_dominant_circles or both min_period and max_period shalle be not null.")
            return None, None, None, None, None



        # Normalize the working index before any date slicing.
        original_data.index = pd.to_datetime(original_data.index)
        original_data = original_data.sort_index()



        # ------------------------------------------------------
        # Select the analysis window ending at current_date
        # ------------------------------------------------------

        # Common notebook path: current_date + num_samples define the window.
        if((current_date != None)  & (num_samples != None)):


            # Types conversion




            if original_data.index.tz is not None:
                # Match current_date to the timezone already used by the data.
                current_date = pd.Timestamp(current_date).tz_convert(original_data.index.tz)
            else:
                # Keep daily/timezone-naive data timezone-naive.
                current_date = pd.Timestamp(current_date).tz_localize(None)
                


            filtered_data_cd = original_data[original_data.index == current_date]
            


            if (filtered_data_cd.empty):

                filtered_data_cd = original_data[original_data.index.date == current_date.date()]

            # Use the latest available timestamp not after current_date.
            if not filtered_data_cd.empty:


                data = original_data[original_data.index <= current_date]
                max_datetime = data.index.max()
                index_of_max_time_for_cd = original_data.index.get_indexer([max_datetime])[0]
                    


                    
                available_n_samples = len(data)
                

                data = data.tail(num_samples)
                
                
                if len(data) != num_samples:
                    raise ValueError(f"Mismatch in selected sample size: expected {num_samples}, got {len(data)}")



                start_rebuilt_signal_index = index_of_max_time_for_cd - num_samples + 1
                end_rebuilt_signal_index = index_of_max_time_for_cd + num_samples
                
                
                if start_rebuilt_signal_index < 0:
                    raise ValueError(f"For the period range from {min_period} to {max_period}, "
                                     f"a total of {num_samples} samples before the current datetime {current_date} "
                                     f"is required, but only {available_n_samples} samples are available.")

                if data.empty:
                    print(f"No data for current datetime equal to {current_date}. Possible not existing date (weekend?), wrong date format or mismatch with the timezone.")
                    
                    sys.exit("No data for current datetime")

                    return None, None, None, None, None

            else:
                print(f"No data for current datetime equal to {current_date}.")
                sys.exit("No data for current datetime")
                return None, None, None, None, None



        # ------------------------------------------------------
        # Prepare frequency grid and detrended input series
        # ------------------------------------------------------


        # Build the normalized frequency search grid from the period bounds.
        if(min_period != None):
            max_frequency = int((num_samples / min_period) * 2)
        else:
            max_frequency = int((num_samples / 5) * 2)

        if(max_period != None):
            transform_precision = 1/max_period
        else:
            max_period = int(num_samples/5)
            transform_precision = 1/max_period

        frequency_range = np.arange(transform_precision, max_frequency, transform_precision)

        # Keep the selected price column as the transform input.
        data = data[data_column_name].values


# Detrending intentionally uses original_data so the chosen cut policy is explicit.

        # Replace NaN values before detrending; several filters propagate them.

        original_data[data_column_name] = original_data[data_column_name].fillna(0)


        if(cut_to_date_before_detrending):
            
            detrending_data = original_data[data_column_name].iloc[:index_of_max_time_for_cd+1]
            print(f'Detrend filter applied to data cut to current datetime: {detrending_data.tail(1).index}')
            print(f'Last available data datetime: {original_data[data_column_name].tail(1).index}')
        else:
            print('Detrend filter applied to full original data time series not cut to current datetime.')
            detrending_data = original_data[data_column_name]

        
        # Select the detrending branch without altering the transform path.
        if(detrend_type == 'linear'):
            print(f'linear detrend, detrend window = {detrend_window}')
            print(f'len orginal_data[data_column_name] = {len(original_data[data_column_name])}')

            detrended_data = self.linear_detrend(detrending_data[data_column_name], window_size = detrend_window)
    

        if(detrend_type == 'quadratic'):
            print('quadratic detrend')           
            detrended_data = detrend(detrending_data, order=2)

        if(detrend_type == 'hp_filter'):
            print('hp_filter')
            detrended_data, _ = self.hp_filter(detrending_data, hp_filter_lambda)


        if(detrend_type == 'jh_filter'):
            print('jh_filter')
            detrended_data = self.jh_filter(detrending_data, jp_filter_p, jp_filter_h)
            
            
        if(detrend_type == 'lowess'):
            print('lowess')
            _, detrended_data = self.detrend_lowess(detrending_data, max_period, k=4)
            
        
        if original_data[data_column_name].isnull().all():
            raise ValueError(f"All values in '{data_column_name}' are NaN — detrending aborted.")



        
        if(time_tracking):
            self.track_time('\tPartial time Data Preparation')


        # ------------------------------------------------------
        # Run Goertzel transform over candidate frequencies
        # ------------------------------------------------------


        # Store amplitude, phase and offset metadata for each candidate.
        harmonics_amplitudes = []
        phases = []
        minoffset = []
        maxoffset = []
        frequency_range = frequency_range/num_samples


        for frequency in frequency_range:

            reduced_detrended_data = detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1]
            
            if(windowing == 'kaiser'):
                window = kaiser(len(reduced_detrended_data), kaiser_beta)
                reduced_detrended_data = reduced_detrended_data * window
            
            
            temp_amplitudes, temp_phases, _, temp_minoffset, temp_maxoffset = goertzel_DFT(
                reduced_detrended_data.values.astype(np.float64), 
                1/frequency
            )
 
            harmonics_amplitudes.append(temp_amplitudes)
            phases.append(temp_phases)
            minoffset.append(temp_minoffset)
            maxoffset.append(temp_maxoffset)

        minoffset = [int(x) for x in minoffset]
        maxoffset = [int(x) for x in maxoffset]


        harmonics_amplitudes = np.array(harmonics_amplitudes)
        phases = np.array(phases)
 
        # Extract local maxima from the amplitude spectrum.
        goertzel_df_peaks = pd.DataFrame()
        peaks_indexes = argrelmax(harmonics_amplitudes, order = 10)[0] # find indexes of peaks

        peak_frequencies = np.array(frequency_range[peaks_indexes])
        peak_periods = np.array(1 / frequency_range[peaks_indexes])
        peak_amplitudes = harmonics_amplitudes[peaks_indexes]
        peak_phases = phases[peaks_indexes]
        peak_next_min_offset = np.array(minoffset)[peaks_indexes]
        peak_next_max_offset = np.array(maxoffset)[peaks_indexes]


        # Frequency-scaled amplitudes are used for dominant-cycle ranking.
        scaled_peak_amplitudes = peak_amplitudes*peak_frequencies
        scaled_harmonics_amplitudes = harmonics_amplitudes*frequency_range

        goertzel_df_peaks['peaks_indexes'] = peaks_indexes
        goertzel_df_peaks['peak_frequencies'] = peak_frequencies
        goertzel_df_peaks['peak_periods'] = peak_periods
        goertzel_df_peaks['peak_amplitudes'] = peak_amplitudes
        goertzel_df_peaks['scaled_peak_amplitudes'] = scaled_peak_amplitudes
        goertzel_df_peaks['peak_phases'] = peak_phases
        goertzel_df_peaks['peak_next_min_offset'] = peak_next_min_offset
        goertzel_df_peaks['peak_next_max_offset'] = peak_next_max_offset




        if(time_tracking):
            self.track_time('\tPartial time Goertzel Transform')
            
            
        # ------------------------------------------------------
        # Restrict harmonic candidates to the requested period range
        # ------------------------------------------------------
        # Keep either an explicit number of harmonics or all peaks inside
        # min_period..max_period.
        cut_peaks_indexes = []


        if(limit_n_harmonics != None):
            cut_peaks_indexes = peaks_indexes[0:limit_n_harmonics]

        elif((min_period != None) & (max_period != None)):


            for index, period in enumerate(peak_periods):

                if((period >= min_period) & (period <= max_period)):
                    cut_peaks_indexes.append(peaks_indexes[index])

        peak_frequencies = frequency_range[cut_peaks_indexes]
        peak_periods = 1 / frequency_range[cut_peaks_indexes]
        scaled_peak_amplitudes = scaled_harmonics_amplitudes[cut_peaks_indexes]
        peak_phases = phases[cut_peaks_indexes] #np.angle(transform[cut_peaks_indexes])

        if(len(cut_peaks_indexes) == 0):
            print(f"\t\t\t\tlen(cut_peaks_indexes) == 0")
            return None, None, None, None, None
        
        


                            
        if(time_tracking):
            self.track_time('\tPartial time Partial time Limit N of harmonics')
        

        # ------------------------------------------------------
        # Apply optional Bartels-style peak filtering
        # -----------------------------------------------------

        temp_filtered_f_indexes = np.array([], dtype=int)
        goertzel_df_peaks['bartel_score'] = np.nan

        if(bartel_peaks_filtering == True):
            for index in cut_peaks_indexes:
                frequency = frequency_range[index]
                cycle_length = 1 / frequency
                divisor = 100 #16
                max_segments = 30 # int(num_samples/divisor)

                bartelsscore, _ = self.get_bartels_score(data, cycle_length, max_segments) #get_bartels_score(detrended_data, cycle_length, max_segments)
                goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'bartel_score'] = bartelsscore

                if(bartelsscore >= bartel_scoring_threshold):
                    temp_filtered_f_indexes = np.append(temp_filtered_f_indexes, int(index))

            dominant_peaks_indexes = temp_filtered_f_indexes
        else:
            dominant_peaks_indexes = peaks_indexes



        if(time_tracking):
            self.track_time('\tPartial time Bartel Score')                           


        # ------------------------------------------------------
        # Compute optional correlation features for peak scoring
        # -----------------------------------------------------

        if(other_correlations == True):

            scaler = StandardScaler()

            data_df = pd.DataFrame()

            scaled_correlations =     []
            tau = 0
            unscaled_pearson_correlation  = []
            scaled_pearson_correlation = []
            scaled_spearman_correlation  = []
            scaled_kendall_correlation   = []

            time = np.linspace(0, num_samples, num_samples, endpoint=True)

            data_df[data_column_name] = data

            for index in cut_peaks_indexes:

                phase = phases[index] # np.angle(transform[index])
                amplitude = harmonics_amplitudes[index] # np.abs(transform[index])
                frequency = frequency_range[index]
                period = 1 / frequency

                signal = pd.DataFrame()
                signal['circle_signal'] = amplitude * np.sin(2 * np.pi * frequency * time + phase)
                signal['scaled_signal'] = scaler.fit_transform(signal['circle_signal'].values.reshape(-1, 1)).flatten()

                averages = pd.DataFrame()


                # Compare the harmonic with the Savitzky-Golay delta proxy.

                if(debug == True):
                    print("Other correlations, period: " + str(period))
                    print("Other correlations, int(period*2): " + str(int(period*2)))
                    print("len(data_df[data_column_name]): " + str(len(data_df[data_column_name])))

                averages['savgol_filter_long'] = savgol_filter(data_df[data_column_name], int(period*2), 2)
                averages['savgol_filter_short'] = savgol_filter(data_df[data_column_name], int(period), 2)
                averages['savgol_filter_delta'] = averages['savgol_filter_short']  - averages['savgol_filter_long']
                averages['scaled_savgol_filter_delta'] = scaler.fit_transform(averages['savgol_filter_delta'].values.reshape(-1, 1)).flatten()



                goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_savgol_filter_delta_correlation'] = simpson(signal['scaled_signal'] * np.roll(averages['scaled_savgol_filter_delta'], tau), dx=1)


                            
                            
            if(time_tracking):
                self.track_time('\tPartial time Other Correlations')      

            # Measure peak-count coherence against the Savitzky-Golay delta proxy.
            if(int(period/2) < 2):
                peaks_tollerance = 1
            else:
                peaks_tollerance = int(period/2)

            signal_max_peaks_indexes = argrelmax(signal['scaled_signal'].values, order = peaks_tollerance)[0] # find indexes of peaks
            signal_min_peaks_indexes = argrelmin(signal['scaled_signal'].values, order = peaks_tollerance)[0] # find indexes of peaks
            signal_peaks_n = len(signal_max_peaks_indexes) + len(signal_min_peaks_indexes)

            scaled_savgol_filter_delta_max_peaks_indexes = argrelmax(averages['scaled_savgol_filter_delta'].values, order = peaks_tollerance)[0] # find indexes of peaks
            scaled_savgol_filter_delta_min_peaks_indexes = argrelmin(averages['scaled_savgol_filter_delta'].values, order = peaks_tollerance)[0] # find indexes of peaks
            scaled_savgol_filter_delta_peaks_n = len(scaled_savgol_filter_delta_max_peaks_indexes) + len(scaled_savgol_filter_delta_min_peaks_indexes)

            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_signal_vs_scaled_savgol_filter_delta_peaks_n_ratio'] = abs(signal_peaks_n - scaled_savgol_filter_delta_peaks_n) / period # peaks_n_ratio
            
                            
            if(time_tracking):
                self.track_time('\tPartial time Partial time Peaks cardinality Error calculation')   


            # Compare the harmonic with the derivative of the delta proxy.
            averages['scaled_savgol_filter_delta_derivate'] = averages.diff()['scaled_savgol_filter_delta'] #averages['scaled_average_delta'].values - averages['scaled_average_delta'].shift(1).values #averages['scaled_average_delta'].diff()
            averages['scaled_savgol_filter_delta_derivate'] = averages['scaled_savgol_filter_delta_derivate'].fillna(0)
            signal['scaled_signal_derivate'] =  signal.diff()['scaled_signal'] #signal['scaled_signal'].values - signal['scaled_signal'].shift(1).values #signal['scaled_signal'].diff()
            signal['scaled_signal_derivate'] = signal['scaled_signal_derivate'] .fillna(0)

            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_savgol_filter_delta_derivate_correlations'] = simpson(signal['scaled_signal_derivate'] * np.roll(averages['scaled_savgol_filter_delta_derivate'], tau), dx=1) / signal_peaks_n
                            
                            
            if(time_tracking):
                self.track_time('\tPartial time Partial time Correlation between scaled harmonic and ascled delta')   


            # Measure nearest-peak phase error between harmonic and proxy.

            # Store paired peak-index differences before RMSE normalization.
            differences = []

            if(len(scaled_savgol_filter_delta_max_peaks_indexes) > 0 and len(signal_max_peaks_indexes) > 0):

                # Use the denser max-peak set as the reference series.
                if(len(scaled_savgol_filter_delta_max_peaks_indexes) >= len(signal_max_peaks_indexes)):
                    peak_indices_series1 = scaled_savgol_filter_delta_max_peaks_indexes
                    peak_indices_series2 = signal_max_peaks_indexes
                else:
                    peak_indices_series1 = signal_max_peaks_indexes
                    peak_indices_series2 = scaled_savgol_filter_delta_max_peaks_indexes

                # Pair each reference max peak with the nearest comparison peak.
                for peak_index_series1 in peak_indices_series1:
                    differences_peak = np.abs(peak_index_series1 - peak_indices_series2)
                    nearest_peak_index = peak_indices_series2[np.argmin(differences_peak)]
                    differences.append(peak_index_series1 - nearest_peak_index)

            # Use the denser min-peak set as the reference series.
            if(len(scaled_savgol_filter_delta_min_peaks_indexes) > 0 and len(signal_min_peaks_indexes) > 0):

                # Select the denser min-peak series.
                if(len(scaled_savgol_filter_delta_min_peaks_indexes) >= len(signal_min_peaks_indexes)):
                    peak_indices_series1 = scaled_savgol_filter_delta_min_peaks_indexes
                    peak_indices_series2 = signal_min_peaks_indexes
                else:
                    peak_indices_series1 = signal_min_peaks_indexes
                    peak_indices_series2 = scaled_savgol_filter_delta_min_peaks_indexes

                # Pair each reference min peak with the nearest comparison peak.
                for peak_index_series1 in peak_indices_series1:
                    differences_peak = np.abs(peak_index_series1 - peak_indices_series2)
                    nearest_peak_index = peak_indices_series2[np.argmin(differences_peak)]
                    differences.append(peak_index_series1 - nearest_peak_index)

            if(len(differences) > 0):
                # Convert peak-index distances to an RMSE score.
                root_mean_square_error = np.sqrt(np.mean(np.array(differences) ** 2))

            else:
                root_mean_square_error = period

            # Normalize the peak-phase error by the cycle period.
            root_mean_square_error = root_mean_square_error / period

            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_signal_vs_scaled_savgol_filter_delta_peaks_phase_RMSE'] = root_mean_square_error


                            
        if(time_tracking):
            self.track_time('\tPartial time error between current scaled harmonic and scaled_savgol_filter_delta_correlation')   

        # ------------------------------------------------------
        # Build the global score used to rank candidate cycles
        # ------------------------------------------------------

        if(bartel_peaks_filtering == True and bartel_scoring_threshold > 0):
            ascending_columns = ['bartel_score',
                              'scaled_savgol_filter_delta_correlation',
                              'scaled_savgol_filter_delta_derivate_correlations',
                              'scaled_peak_amplitudes'] # Higher score for higher values
        else:
            ascending_columns = ['scaled_savgol_filter_delta_correlation',
                              'scaled_savgol_filter_delta_derivate_correlations',
                              'scaled_peak_amplitudes'] # Higher score for higher values


        descending_columns = ['scaled_signal_vs_scaled_savgol_filter_delta_peaks_phase_RMSE',
                              'scaled_signal_vs_scaled_savgol_filter_delta_peaks_n_ratio'] # Higer score for lower values

        if 'scaled_savgol_filter_delta_correlation' not in goertzel_df_peaks:
            print(goertzel_df_peaks)

        global_score = self.get_gloabl_score(goertzel_df_peaks, ascending_columns, descending_columns)

        goertzel_df_peaks['global_score'] = global_score

        
        if(time_tracking):
            self.track_time('\tPartial time Cicle Global Scoring calculation')   

        # ------------------------------------------------------
        # Select the dominant peaks used for signal reconstruction
        # -----------------------------------------------------

        dominant_peak_frequencies = frequency_range[dominant_peaks_indexes]
        dominant_peak_periods = 1 / frequency_range[dominant_peaks_indexes]
        dominant_peak_amplitudes = harmonics_amplitudes[dominant_peaks_indexes]
        dominant_scaled_peak_amplitudes = scaled_harmonics_amplitudes[dominant_peaks_indexes]
        dominant_peak_phases = phases[dominant_peaks_indexes] # np.angle(transform[dominant_peaks_indexes])
        dominant_peak_next_min_offset = np.array(minoffset)[dominant_peaks_indexes]
        dominant_peak_next_max_offset = np.array(maxoffset)[dominant_peaks_indexes]

        dominant_peaks_global_score = []
        for index in dominant_peaks_indexes: #goertzel_df_peaks[goertzel_df_peaks['peaks_indexes'] == dominant_peaks_indexes]['global_score']
            dominant_peaks_global_score.append(goertzel_df_peaks[goertzel_df_peaks['peaks_indexes'] == index]['global_score'].values[0])

        dominant_peaks = pd.DataFrame()
        dominant_peaks['dominant_peaks_indexes'] = dominant_peaks_indexes
        dominant_peaks['peak_frequencies'] = dominant_peak_frequencies
        dominant_peaks['peak_periods'] = dominant_peak_periods
        dominant_peaks['peak_amplitudes'] = dominant_peak_amplitudes
        dominant_peaks['scaled_peak_amplitudes'] = dominant_scaled_peak_amplitudes
        dominant_peaks['peak_phases'] = dominant_peak_phases
        dominant_peaks['peak_next_min_offset'] = dominant_peak_next_min_offset
        dominant_peaks['peak_next_max_offset'] = dominant_peak_next_max_offset
        dominant_peaks['global_score'] = dominant_peaks_global_score


        if(dominant_cicles_sorting_type == 'global_score'):
            sorted_indices = dominant_peaks['global_score'].argsort()[::-1]
            sorted_dominant_peaks_indexes = dominant_peaks['dominant_peaks_indexes'].iloc[sorted_indices]

        else:
            sorted_indices = dominant_peaks['scaled_peak_amplitudes'].argsort()[::-1]
            sorted_dominant_peaks_indexes = dominant_peaks['dominant_peaks_indexes'].iloc[sorted_indices]

        used_indexes = sorted_dominant_peaks_indexes[0:final_kept_n_dominant_circles]


        # Keep only the selected dominant-cycle indexes.
        kept_dominant_peak_frequencies = frequency_range[used_indexes]
        kept_dominant_peak_periods = 1 / frequency_range[used_indexes]
        kept_dominant_peak_amplitudes = harmonics_amplitudes[used_indexes]
        kept_dominant_scaled_peak_amplitudes = scaled_harmonics_amplitudes[used_indexes]
        kept_dominant_peak_phases = phases[used_indexes] # np.angle(transform[used_indexes])

        kept_dominant_peaks = pd.DataFrame()
        kept_dominant_peaks['dominant_peaks_indexes'] = used_indexes
        kept_dominant_peaks['peak_frequencies'] = kept_dominant_peak_frequencies
        kept_dominant_peaks['peak_periods'] = kept_dominant_peak_periods
        kept_dominant_peaks['peak_amplitudes'] = kept_dominant_peak_amplitudes
        kept_dominant_peaks['scaled_peak_amplitudes'] = kept_dominant_scaled_peak_amplitudes
        kept_dominant_peaks['peak_phases'] = kept_dominant_peak_phases


        if(time_tracking):
            self.track_time('\tPartial time Dominants Peaks Sorting')   

        # ------------------------------------------------------
        # Derive the dominant period used by indicator proxy columns
        # ------------------------------------------------------

        rebuilt_sig_left_zeros = np.zeros(start_rebuilt_signal_index )

        max_period_dominant_circle = 0
        for index in used_indexes:

            temp_period = 1 / frequency_range[index]

            if(max_period_dominant_circle < temp_period):
                max_period_dominant_circle = temp_period


        max_dominant_peak_period = 0
        max_dominant_peak_scaled_amplitude = 0
        for index in used_indexes:

            frequency = frequency_range[index]

            if(max_dominant_peak_scaled_amplitude < scaled_harmonics_amplitudes[index]):
                max_dominant_peak_period = 1 / frequency
                max_dominant_peak_scaled_amplitude = scaled_harmonics_amplitudes[index]


        dominant_period = max_period_dominant_circle #max_dominant_peak_period

        
            
        if(time_tracking):
            self.track_time('\tPartial time Dominant Circle Calibrated Standard Indicator')   
            
        # ------------------------------------------------------
        # Build centered moving-average delta proxy
        # -----------------------------------------------------

        data_subset_for_average = original_data[start_rebuilt_signal_index:index_of_max_time_for_cd]

        long_average = data_subset_for_average[data_column_name].rolling(window=round(dominant_period), center=centered_averages).mean()
        short_average = data_subset_for_average[data_column_name].rolling(window=round(dominant_period/2), center=centered_averages).mean()

        centered_averages_delta = -(long_average - short_average)
        centered_averages_delta = np.concatenate((rebuilt_sig_left_zeros, centered_averages_delta), axis = 0)

        average_delta_right_zeros = np.zeros(len(original_data) - len(centered_averages_delta))
        centered_averages_delta = np.concatenate((centered_averages_delta, average_delta_right_zeros), axis = 0)
        original_data['centered_averages_delta'] = centered_averages_delta

            
        if(time_tracking):
            self.track_time('\tPartial time Averages Delta')   
            


        # ------------------------------------------------------
        # Build Savitzky-Golay delta proxy
        # -----------------------------------------------------


        long_scaled_savgol_filter = savgol_filter(data_df[data_column_name], int(dominant_period*2), 2)
        short_scaled_savgol_filter = savgol_filter(data_df[data_column_name], int(dominant_period), 2)


        scaled_savgol_filter_delta = short_scaled_savgol_filter - long_scaled_savgol_filter
        scaled_savgol_filter_delta = np.concatenate((rebuilt_sig_left_zeros, scaled_savgol_filter_delta), axis = 0)

        average_delta_right_zeros = np.zeros(len(original_data) - len(scaled_savgol_filter_delta))
        scaled_savgol_filter_delta = np.concatenate((scaled_savgol_filter_delta, average_delta_right_zeros), axis = 0)
        original_data['scaled_savgol_filter_delta'] = scaled_savgol_filter_delta

            
        if(time_tracking):
            self.track_time('\tPartial time scaled_savgol_filter_delta') 

            
        # ----------------------------------------------------------
        # Rebuild individual dominant cycles and their composite signal
        # ----------------------------------------------------------

        time = np.linspace(0, num_samples*2, num_samples*2, endpoint=False)
        dominant_circle_signal = np.zeros(len(time), dtype=float)
        composite_dominant_cycle_signal = np.zeros(len(time), dtype=float)

        max_dominant_peak_period = 0
        max_dominant_peak_scaled_amplitude = 0
        max_period_dominant_circle = 0

        signals_results['start_rebuilt_signal_index'] = start_rebuilt_signal_index


        signals_results['end_rebuilt_signal_index'] = end_rebuilt_signal_index

        signals_results['dominant_peaks_signals'] = [None] * len(signals_results)
 
        signals = []
        new_columns = {}
        max_extension = 0

        for index in used_indexes:

            phase = phases[index]
            amplitude = harmonics_amplitudes[index]
            frequency = frequency_range[index]
            period = 1 / frequency
            indicators_period = period / 2

            temp_circle_signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
            composite_dominant_cycle_signal += temp_circle_signal

            signal, extension_periods = self.rebuilt_signal_zeros(signal = temp_circle_signal,
                                               start_rebuilt_signal_index = start_rebuilt_signal_index,
                                               data = original_data)      

            if(extension_periods > 0):
                original_data = self.datetime_dateset_extend(original_data, extension_periods)


            new_columns['dominant_circle_signal_period_' + str(period)] = signal

            peak_data = {
                'peak_frequencies': frequency,
                'peak_periods': period,
                'peak_amplitudes': amplitude,
                'peak_phases': phase,
                'sin_dominant_circle_signal_name': 'dominant_circle_signal_period_' + str(period),
                'start_rebuilt_signal_index': int(start_rebuilt_signal_index),
                'end_rebuilt_signal_index': int(end_rebuilt_signal_index)

            }


            signals.append(peak_data)

            if(max_dominant_peak_scaled_amplitude < scaled_harmonics_amplitudes[index]):
                max_dominant_peak_period = 1 / frequency
                max_dominant_peak_scaled_amplitude = scaled_harmonics_amplitudes[index]

            if(max_period_dominant_circle < period):
                max_period_dominant_circle = period


        signals_results['dominant_peaks_signals'] = signals
        

        # Rebuild the combined dominant-cycle signal on the same horizon.
        signal, extension_periods = self.rebuilt_signal_zeros(signal = composite_dominant_cycle_signal,
                                           start_rebuilt_signal_index = start_rebuilt_signal_index,
                                           data = original_data)

        if(extension_periods > 0):
            original_data = self.datetime_dateset_extend(original_data, extension_periods)


        original_data['composite_dominant_circles_signal'] = signal
        new_columns['composite_dominant_circles_signal'] = signal


            
        if(time_tracking):
            self.track_time('\tPartial time Dominant Cicle Signals') 



        # ------------------------------------------------------
        # Attach detrended and rebuilt columns back to original_data
        # ------------------------------------------------------


        new_columns['detrended'] = detrended_data


        # Use original_data as the alignment length for generated columns.
        max_len = len(original_data)
        
        # Pad shorter generated columns with NaN before dataframe assembly.
        for key in new_columns:

            col_len = len(new_columns[key])
            if col_len < max_len:

                add_len = max_len - col_len
        
                # Create a Series with NaNs to append.
                new_nans = pd.Series([np.nan] * add_len, index=range(col_len, max_len))

                # Use pd.concat to append NaNs to the existing Series.
                new_columns[key] = pd.concat([new_columns[key], new_nans])        


        new_data = pd.DataFrame(new_columns)   

        # Align generated rows to original_data using the first generated index.
        first_date = pd.to_datetime(new_data.index[0], errors='coerce')

        # Match timezone handling to original_data before locating rows.
        if original_data.index.tz is None:
            first_date = first_date.tz_localize(None)
        else:
            first_date = first_date.tz_convert(original_data.index.tz)

        # Copy generated columns into the matching original_data slice.
        if first_date in original_data.index:
            start_index = original_data.index.get_loc(first_date)
            end_index = min(start_index + len(new_data), len(original_data))

            # Add generated columns that do not exist yet.
            missing_cols = [col for col in new_data.columns if col not in original_data.columns]
            for col in missing_cols:
                original_data[col] = np.nan

            # Reindex generated data onto the expected target rows.
            matching_rows = original_data.index[start_index:end_index]
            available_rows = matching_rows.intersection(new_data.index)
            new_data_aligned = new_data.reindex(available_rows)

            # Preserve all expected rows even when some generated rows are absent.
            new_data_final = pd.DataFrame(index=matching_rows, columns=new_data.columns)
            new_data_final.update(new_data_aligned)

            # Apply only rows that are present in original_data.
            existing_rows = matching_rows.intersection(original_data.index)

            if not existing_rows.empty:
                for col in new_data.columns:
                    if col != 'composite_dominant_circles_signal':
                        original_data.loc[existing_rows, col] = pd.to_numeric(new_data_final.loc[existing_rows, col].values, errors='coerce')


        else:
            print("La data della prima riga di new_data non è presente in original_data.")


            
        if(time_tracking):
            self.track_time('\tPartial time Detrended Data') 


        # ------------------------------------------------------
        # Refresh the Savitzky-Golay delta column after alignment
        # -----------------------------------------------------

        long_scaled_savgol_filter = savgol_filter(data_df[data_column_name], int(dominant_period*2), 2)
        short_scaled_savgol_filter = savgol_filter(data_df[data_column_name], int(dominant_period), 2)

        scaled_savgol_filter_delta = short_scaled_savgol_filter - long_scaled_savgol_filter
        scaled_savgol_filter_delta = np.concatenate((rebuilt_sig_left_zeros, scaled_savgol_filter_delta), axis = 0)

        average_delta_right_zeros = np.zeros(len(original_data) - len(scaled_savgol_filter_delta))
        scaled_savgol_filter_delta = np.concatenate((scaled_savgol_filter_delta, average_delta_right_zeros), axis = 0)
        original_data['scaled_savgol_filter_delta'] = scaled_savgol_filter_delta
        
        
        df = pd.DataFrame(signals_results)
        
            
        if(time_tracking):
            self.track_time('\tPartial time scaled_savgol_filter_delta') 



        # ------------------------------------------------------
        # Optional notebook report
        # ------------------------------------------------------

        if(print_report == True):
            display(configuration)



            print("\n")
            display(goertzel_df_peaks)

            print("\n")
            display(kept_dominant_peaks)



        # ------------------------------------------------------
        # Optional Plotly diagnostics
        # ------------------------------------------------------


        if(show_charts == True):

            # Spectrum chart.
            basic_rows_n = 2

            # Plot scaled peak amplitudes across detected periods.
            spectrum_trace = go.Scatter(x=frequency_range, y=harmonics_amplitudes, mode='lines', name='Goetzel DFT Spectrum')
            fig_spectrum = go.Figure(spectrum_trace)
            fig_spectrum.update_layout(title="Frequency Spectrum", xaxis=dict(title="Frequency"), yaxis=dict(title="Magnitude"))

            fig_spectrum.show()


            # Price, detrended signal and dominant-cycle reconstruction.
            fig = make_subplots(rows=basic_rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Original Data", "Detrended Data", "Dominant Circles Signal", "Centered Averages Delta"))


            fig.add_trace(go.Scatter(x=original_data.index,
                                      y=original_data[data_column_name],
                                      mode="lines",
                                      name="Original data"),
                        row=1,
                        col=1)

            # Mark the analysis anchor date.
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=original_data[data_column_name].min(),
                y1=original_data[data_column_name].max(),
                line=dict(color='purple', width=1),
                row=1, col=1
            )
            
            # Normalize the composite signal for visual comparison.
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_detrended = scaler.fit_transform(original_data['detrended'].values.reshape(-1, 1)).flatten()
            normalized_composite_circles = scaler.fit_transform(original_data['composite_dominant_circles_signal'].values.reshape(-1, 1)).flatten()

            print(f"original data last element datetime: {original_data[data_column_name].last_valid_index()}") 
            print(f"detrended data last element datetime: {original_data['detrended'].last_valid_index()}") 
            print(f"CDC data last element datetime: {original_data['composite_dominant_circles_signal'].last_valid_index()}") 

            fig.add_trace(go.Scatter(x=original_data.index, y=normalized_detrended, mode="lines", name="Detrended Close"), row=2, col=1)
            fig.add_trace(go.Scatter(x=original_data.index, y=normalized_composite_circles, mode="lines", name="Dominant Circle Signal"), row=2, col=1)





            # Mark the analysis anchor date on the reconstruction subplot.
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=-1,
                y1=+1,
                line=dict(color='purple', width=1),
                row=2, col=1
            )


            fig.update_layout(title="Goertzel Dominant Cyrcles Analysis", height=800)


            fig.update_xaxes(type="category")

            # Display the diagnostic subplot figure.
            fig.show()



        return current_date, index_of_max_time_for_cd, original_data, signals_results, configuration
