"""Multi-period cycle reconstruction workflow for cyPredict."""

import multiprocessing
import random

from deap import algorithms, base, creator, tools
from hyperopt import atpe, fmin, hp, tpe
from hyperopt.pyll import scope
from IPython.display import display
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ..native_imports import (
    REQUIRED_CYGAOPT_ABI_VERSION,
    ensure_native_module_paths,
    require_native_abi,
)

ensure_native_module_paths()

import cyGAopt
import cyGAoptMultiCore
from cyGAopt import run_genetic_algorithm
from cyGAoptMultiCore import run_genetic_algorithm as run_genetic_algorithm_multicore
from goertzel import goertzel_DFT

require_native_abi(cyGAopt, "cyGAopt", REQUIRED_CYGAOPT_ABI_VERSION)
require_native_abi(cyGAoptMultiCore, "cyGAoptMultiCore", REQUIRED_CYGAOPT_ABI_VERSION)


class MultiperiodMixin:
    """Run and refit multiple dominant-cycle period ranges."""

    def multiperiod_analysis_from_config(self, config):
        """Run ``multiperiod_analysis`` from a ``MultiPeriodAnalysisConfig``.

        The method only expands structured configuration into the legacy
        keyword signature, so existing calculation behavior is unchanged.
        """
        return self.multiperiod_analysis(**config.to_legacy_kwargs())

    def multiperiod_analysis_result(self, *args, **kwargs):
        """Run ``multiperiod_analysis`` and return a ``MultiPeriodResult``."""
        from ..results import MultiPeriodResult

        return MultiPeriodResult.from_legacy_tuple(self.multiperiod_analysis(*args, **kwargs))

    def multiperiod_analysis(self,
                             data_column_name,
                             current_date,
                             periods_pars,
                             best_fit_start_back_period = None,                             
                             show_charts = True,
                             population_n = 40,
                             CXPB = 0.7,
                             MUTPB = 0.3,
                             NGEN = 100,
                             MultiAn_fitness_type = "mse",
                             MultiAn_fitness_type_svg_smoothed = True,
                             MultiAn_fitness_type_svg_filter = 5,
                             weigth = -1.0,
                             reference_detrended_data = "longest", # less_detrended
                             windowing = None,
                             kaiser_beta = 5,
                             enable_cycles_alignment_analysis = True,
                             opt_algo_type = 'genetic_omny_frequencies',  # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency'
                             amplitudes_inizialization_type = "random",
                             frequencies_ft = True,
                             phases_ft = True,
                             cut_to_date_before_detrending = True,
                             detrend_type = 'hp_filter',
                             lowess_k = 3,
                             linear_filter_window_size_multiplier = 1,
                             period_related_rebuild_range = False,
                             period_related_rebuild_multiplier = 2.5,
                             discretization_steps = 1000,
                             enabled_multiprocessing = True,
                             log_level = None,
                             log_to_console = None,
                             log_to_file = None,
                             log_dir = None,
                             log_run_id = None,
                             random_seed = None,
                            ):
        """Run multiple period ranges and refit the combined cycle signal.

        ``multiperiod_analysis`` decomposes one analysis date into several
        period ranges, calls ``analyze_and_plot`` for each row of
        ``periods_pars``, merges the selected dominant cycles, and then
        optimizes/refits amplitudes, frequencies and phases according to
        ``opt_algo_type``. This is the main workflow used by the notebooks for
        projected local extrema and for downstream min/max analysis.

        Parameters
        ----------
        data_column_name : str
            Price column to analyze, normally ``"Close"``.
        current_date : str or datetime-like
            Last observed bar used as the analysis anchor. Future projections
            are built after this point.
        periods_pars : pandas.DataFrame
            Range definition table. Required columns are
            ``num_samples``, ``final_kept_n_dominant_circles``,
            ``min_period``, ``max_period`` and ``hp_filter_lambda``. Optional
            row-level columns such as ``detrend_type`` and ``lowess_k`` override
            method-level defaults for that range.
        best_fit_start_back_period : int, optional
            Legacy tuning argument stored on ``self`` and used by selected
            evaluation paths. Some callers pass it while using algorithms where
            it is not active.
        show_charts : bool, default True
            Enables Plotly chart creation and display.
        population_n, CXPB, MUTPB, NGEN : int or float
            Genetic optimizer population size, crossover probability, mutation
            probability and number of generations. Meaningful for genetic
            ``opt_algo_type`` values.
        MultiAn_fitness_type : str, default "mse"
            Fitness metric selector used by the multi-analysis optimizer.
        MultiAn_fitness_type_svg_smoothed : bool, default True
            If true, smooths the reference detrended series before optimization.
        MultiAn_fitness_type_svg_filter : int, default 5
            Savitzky-Golay filter window used when smoothing is enabled.
        weigth : float, default -1.0
            DEAP fitness weight. Negative values minimize loss.
        reference_detrended_data : {"longest", "less_detrended"}
            Selects which single-range detrended series becomes the optimizer
            reference. ``"longest"`` uses the range with most samples.
            ``"less_detrended"`` chooses the least aggressive detrend proxy
            for HP/LOWESS paths.
        windowing, kaiser_beta
            Routed to ``analyze_and_plot``. ``kaiser_beta`` is meaningful only
            with Kaiser windowing.
        enable_cycles_alignment_analysis : bool, default True
            Enables cycle-alignment KPI calculations.
        opt_algo_type : str, default "genetic_omny_frequencies"
            Optimizer branch. Known values in the implementation include
            ``"mono_frequency"``, ``"genetic_omny_frequencies"``,
            ``"genetic_frequencies_ranges"``, ``"tpe"`` and ``"atpe"``.
            ``frequencies_ft`` and ``phases_ft`` matter only for branches that
            optimize those dimensions.
        amplitudes_inizialization_type : str, default "random"
            Amplitude initialization strategy used by the optimizer.
        frequencies_ft, phases_ft : bool
            Allow optimizer movement around Goertzel frequencies/phases when
            the selected algorithm supports those variables.
        cut_to_date_before_detrending : bool, default True
            Passed to single-range analysis to prevent future data leakage in
            detrending.
        detrend_type, lowess_k, linear_filter_window_size_multiplier
            Detrending controls routed to ``analyze_and_plot``. ``lowess_k`` is
            meaningful only for LOWESS. ``linear_filter_window_size_multiplier``
            sets the linear detrend window as a multiple of ``max_period``.
        period_related_rebuild_range, period_related_rebuild_multiplier
            Control whether rebuilt-cycle comparison windows are constrained by
            period length.
        discretization_steps : int, default 1000
            Discrete grid size used in segmented mutation.
        enabled_multiprocessing : bool, default True
            Enables multiprocessing in supported optimizer branches. Nested
            callers often disable this to avoid double multiprocessing.
        log_level, log_to_console, log_to_file, log_dir, log_run_id : optional
            Structured logging override for this analysis call. These values
            update the instance logger before the workflow starts. Leave them
            unset to keep the constructor logging configuration.
        random_seed : int, optional
            Seed passed to Python and C++ stochastic optimizers. Leave unset to
            keep non-deterministic genetic runs.
        Returns
        -------
        tuple
            Existing notebook contract:
            ``(elaborated_data_df, signals_results_df, composite_signal,
            configurations, bb_delta, cdc_rsi, index_of_max_time_for_cd,
            scaled_signals, best_fitness_value)``.

            ``elaborated_data_df`` and ``signals_results_df`` concatenate
            single-range outputs. ``composite_signal`` is the refit projected
            dominant-cycle signal. ``scaled_signals`` contains scaled alignment
            and weighted KPI outputs when enabled.

        Example
        -------
        >>> periods = pd.DataFrame(
        ...     [[256, 4, 10, 80, 1600]],
        ...     columns=["num_samples", "final_kept_n_dominant_circles",
        ...              "min_period", "max_period", "hp_filter_lambda"])
        >>> result = cp.multiperiod_analysis(
        ...     data_column_name="Close",
        ...     current_date="2023-12-29",
        ...     periods_pars=periods,
        ...     opt_algo_type="genetic_omny_frequencies",
        ...     show_charts=False,
        ...     enabled_multiprocessing=False)
        """

        if any(
            value is not None
            for value in (log_level, log_to_console, log_to_file, log_dir, log_run_id)
        ):
            self.configure_logging(
                log_level=log_level,
                log_to_console=log_to_console,
                log_to_file=log_to_file,
                log_dir=log_dir,
                log_run_id=log_run_id,
            )

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.log_info(
            "multiperiod_analysis started",
            function="multiperiod_analysis",
            data_column_name=data_column_name,
            windowing=windowing,
            kaiser_beta=kaiser_beta,
            opt_algo_type=opt_algo_type,
            random_seed=random_seed,
        )
        scaler = self.scaler
        
        self.windowing = windowing
        self.kaiser_beta = kaiser_beta

        self.amplitudes_inizialization_type = amplitudes_inizialization_type
        self.frequencies_ft = frequencies_ft
        self.phases_ft = phases_ft
        
        self.MultiAn_fitness_type = MultiAn_fitness_type
        self.best_fit_start_back_period = best_fit_start_back_period
        self.period_related_rebuild_range = period_related_rebuild_range
        self.period_related_rebuild_multiplier = period_related_rebuild_multiplier

        self.discretization_steps = discretization_steps
        
        self.population_n = population_n
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = NGEN
        self.MultiAn_fitness_type_svg_smoothed = MultiAn_fitness_type_svg_smoothed
        self.MultiAn_fitness_type_svg_filter = MultiAn_fitness_type_svg_filter
        self.weigth = weigth
        self.reference_detrended_data = reference_detrended_data
        self.enable_cycles_alignment_analysis = enable_cycles_alignment_analysis
        self.opt_algo_type = opt_algo_type
        self.cut_to_date_before_detrending = cut_to_date_before_detrending
        self.detrend_type = detrend_type
        self.lowess_k = lowess_k
        self.linear_filter_window_size_multiplier = linear_filter_window_size_multiplier


        elaborated_data_series = [] # pd.DataFrame()
        signals_results_series = [] # pd.DataFrame()
        configurations_series = []
        goertzel_amplitudes = []
        
        if(detrend_type != 'linear' and detrend_type != 'lowess'):
            detrend_type = 'hp_filter'


        self.log_debug(
            "multiperiod_analysis parameters",
            function="multiperiod_analysis",
            parameters={k: v for k, v in locals().items() if k != "self"},
        )
        self.log_timing('1. multiperiod_analysis: entering for loop, start calling analyze_and_plot', function="multiperiod_analysis")
            

        # Optionally reduce the dataframe before single-range detrending.
        original_data = self.data  # non modificato
            
        if cut_to_date_before_detrending:
            max_range_row = periods_pars.loc[periods_pars['max_period'].idxmax()]
            protective_length = int(1.5 * max_range_row['num_samples'] )
        
            valid_times = original_data.index[original_data.index <= current_date]
            if len(valid_times) == 0:
                raise ValueError("Nessun dato disponibile prima di current_date nel dataset.")
            
            last_valid_time = valid_times.max()
            idx_max_time = original_data.index.get_indexer([last_valid_time])[0]
            start_idx = max(0, idx_max_time - protective_length)
        
            reduced_data = original_data.iloc[start_idx:]
            self.log_info(
                "Protective data cut applied",
                function="multiperiod_analysis",
                samples=len(reduced_data),
                start_index=start_idx,
                end_index=idx_max_time,
                start_date=reduced_data.index[0],
                end_date=reduced_data.index[-1],
                current_date=current_date,
            )
        else:
            reduced_data = original_data
            self.log_info(
                "Original dataset not reduced",
                function="multiperiod_analysis",
                samples=len(reduced_data),
                start_index=start_idx,
                end_index=idx_max_time,
            )

            
        
        for index, row in periods_pars.iterrows():

            num_samples = row['num_samples']
            final_kept_n_dominant_circles = row['final_kept_n_dominant_circles']
            min_period = row['min_period']
            max_period = row['max_period']
            hp_filter_lambda = row['hp_filter_lambda']
            
            if('detrend_type' in row.index and row['detrend_type'] is not None):
                detrend_type = row['detrend_type']
            elif detrend_type is not None:
                detrend_type = detrend_type
            else:
                detrend_type = 'hp_filter'

            
            hp_filter_lambda = 1600 # not null default value
            
            if(detrend_type == 'hp_filter'):
                hp_filter_lambda = row['hp_filter_lambda']
                
            if(detrend_type == 'lowess' and row.get('lowess_k') is not None and row['lowess_k'] is not None):
                lowess_k = row['lowess_k']
            else:
                lowess_k = lowess_k
                
             
            self.log_info(
                "Started period-range analysis",
                function="multiperiod_analysis",
                min_period=min_period,
                max_period=max_period,
                num_samples=num_samples,
            )
            if self.is_log_enabled("DEBUG"):
                display(row)

            max_length_series = 0
            max_length_series_index = -1
            


            _, index_of_max_time_for_cd, elaborated_data, signal_results, configuration = self.analyze_and_plot(
                             data = reduced_data, #self.data,
                             data_column_name = data_column_name,
                             num_samples= num_samples,
                             start_date= None,
                             current_date = current_date,
                             final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                             dominant_cicles_sorting_type = 'global_score', #global_score
                             limit_n_harmonics = None,
                             min_period = min_period,
                             max_period = max_period,
                             detrend_type = detrend_type,
                             lowess_k = lowess_k,
                             detrend_window = int(max_period*linear_filter_window_size_multiplier),
                             bartel_scoring_threshold = 0,
                             hp_filter_lambda = hp_filter_lambda,
                             jp_filter_p = 3,
                             jp_filter_h = 100,
                             bartel_peaks_filtering = True,
                             windowing = windowing,
                             kaiser_beta = kaiser_beta,
                             cut_to_date_before_detrending = cut_to_date_before_detrending,
                             centered_averages = True,
                             other_correlations = True,
                             show_charts = False,
                             print_report = False,
                            )
        
            
            elaborated_data_series.append(elaborated_data) # pd.concat([elaborated_data_df, elaborated_data], ignore_index=True)
            signals_results_series.append(signal_results) # pd.concat([signals_results_df, signal_results], ignore_index=True)
            configurations_series.append(configuration)

            if(len(elaborated_data) > max_length_series):

                max_length_series = len(elaborated_data)
                max_length_series_index = index


        # ---------------------------------------------------------------
        # Build the multi-range reference signal and cycle table
        # ---------------------------------------------------------------
        self.log_debug("Starting cycle amplitude refactorization", function="multiperiod_analysis")
        self.log_timing('\n2. multiperiod_analysis: starting Re-Factorize Cyrcles Amplitude', function="multiperiod_analysis")

        # Select the reference detrended series for optimizer fitness.
        if(reference_detrended_data == "less_detrended"):
            if(detrend_type == 'hp_filter'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['hp_filter_lambda'])
            if(detrend_type == 'lowess'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: (configurations_series[i]['lowess_k'] * configurations_series[i]['max_period']))

        # Alternative reference: use the longest single-range analysis.
        if(reference_detrended_data == "longest"):
            index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['num_samples'])
            
        self.log_debug(
            "Selected reference detrended series",
            function="multiperiod_analysis",
            index_detrended_data=index_detrended_data,
        )

        self.MultiAn_reference_detrended_data = elaborated_data_series[index_detrended_data]['detrended'][0:index_of_max_time_for_cd+1]
        

        if(MultiAn_fitness_type_svg_smoothed == True):
            self.MultiAn_reference_detrended_data = savgol_filter(self.MultiAn_reference_detrended_data, MultiAn_fitness_type_svg_filter, 2)

        self.MultiAn_reference_detrended_data = scaler.fit_transform( self.MultiAn_reference_detrended_data.values.reshape(-1, 1)).flatten() * 100


        # Store scaled detrended bounds for optimizer amplitude limits.
        self.MultiAn_detrended_max = np.int64(self.MultiAn_reference_detrended_data.max())
        self.MultiAn_detrended_min = np.int64(self.MultiAn_reference_detrended_data.min())
        
        

        
        # Reset the dominant-cycle table so previous runs cannot leak in.
        self.MultiAn_dominant_cycles_df = pd.DataFrame({
            'peak_frequencies': pd.Series(dtype='float64'),
            'peak_periods': pd.Series(dtype='float64'),
            'peak_phases': pd.Series(dtype='float64'),
            'start_rebuilt_signal_index': pd.Series(dtype='int64'),
            'end_rebuilt_signal_index': pd.Series(dtype='int64')
        })

        max_start_rebuilt_signal_index = 0
        for row in signals_results_series:

            dominant_peaks_signals_list = row['dominant_peaks_signals']

            for structure in dominant_peaks_signals_list:

                peak_frequencies = structure['peak_frequencies']
                peak_amplitudes = structure['peak_amplitudes']
                peak_periods = structure['peak_periods']
                peak_phases = structure['peak_phases']
                start_rebuilt_signal_index = int(structure['start_rebuilt_signal_index'])
                end_rebuilt_signal_index = int(structure['end_rebuilt_signal_index'])

                if(max_start_rebuilt_signal_index < start_rebuilt_signal_index):
                    max_start_rebuilt_signal_index = start_rebuilt_signal_index

                df_row = pd.DataFrame({
                    'peak_frequencies': [peak_frequencies],
                    'multirange_peak_amplitudes': [peak_amplitudes],
                    'peak_periods': [peak_periods],
                    'peak_phases': [peak_phases],
                    'start_rebuilt_signal_index': [start_rebuilt_signal_index],
                    'end_rebuilt_signal_index': [end_rebuilt_signal_index]
                })


                self.MultiAn_dominant_cycles_df = pd.concat([self.MultiAn_dominant_cycles_df, df_row], ignore_index=True)
                
            
        
        # Sort long periods first; several refit branches rely on this order.
        self.MultiAn_dominant_cycles_df = self.MultiAn_dominant_cycles_df.sort_values(by='peak_periods', ascending=False)

        cycles_n = len(self.MultiAn_dominant_cycles_df )
        detrended_abs_max = abs(self.MultiAn_detrended_max - self.MultiAn_detrended_min)
        up_series = pd.Series([detrended_abs_max] * cycles_n)
        low_series = pd.Series([0] * cycles_n)
        
        self.log_timing('3. multiperiod_analysis: end Re-Factorize Cyrcles Amplitude', function="multiperiod_analysis")
        
        

        # -------------------------------------------------------
        # Re-estimate amplitudes with Goertzel on the reference signal
        # -------------------------------------------------------
        
        self.log_timing('\n4. multiperiod_analysis: starting Goertzel best amplitudes', function="multiperiod_analysis")

        goertzel_best_refactoring_df = pd.DataFrame({
            'peak_periods': pd.Series(dtype='float64'),
            'peak_frequencies': pd.Series(dtype='float64'),
            'peak_phases': pd.Series(dtype='float64'),
            'peak_amplitudes': pd.Series(dtype='float64')
        })
        new_rows = []


        for _, row in self.MultiAn_dominant_cycles_df.iterrows():
            
            
            amplitude, phase, _, _, _ = goertzel_DFT(self.MultiAn_reference_detrended_data[int(row['start_rebuilt_signal_index']):int(row['end_rebuilt_signal_index'])], row['peak_periods'])

            new_rows.append({
                'peak_periods': row['peak_periods'],
                'peak_frequencies': 1/row['peak_periods'],
                'peak_phases': phase,
                'peak_amplitudes': amplitude
            })

        goertzel_best_refactoring_df = pd.concat([goertzel_best_refactoring_df, pd.DataFrame(new_rows)], ignore_index=True)
        goertzel_best_refactoring_df = pd.merge(goertzel_best_refactoring_df, 
                                                self.MultiAn_dominant_cycles_df[['peak_periods', 
                                                                                 'start_rebuilt_signal_index', 
                                                                                 'end_rebuilt_signal_index']], 
                                                on='peak_periods', 
                                                how='left')
        goertzel_amplitudes = goertzel_best_refactoring_df['peak_amplitudes']
        self.goertzel_amplitudes = goertzel_amplitudes
        self.MultiAn_dominant_cycles_df["single_range_goertzel_peak_amplitudes"] = goertzel_best_refactoring_df['peak_amplitudes']
        
        self.log_timing('\n5. multiperiod_analysis: end Goertzel best amplitudes', function="multiperiod_analysis")
        
        
        
        # -------------------------------------------------------------------
        # Amplitude-only sweep, processing slower cycles first
        # -------------------------------------------------------------------
        if(opt_algo_type == 'mono_frequency'):
            
            scaler = self.scaler

            amplitudes = []

            temp_circle_signal = []
            len_series = len(self.MultiAn_reference_detrended_data)
            composite_dominant_cycle_signal = pd.Series([0.0] * len_series)
            error = 10 ** 9
            last_error = 10 ** 9
            best_error = 10 ** 9

           
            
            # Iterate periods in descending order.
            for index, row in self.MultiAn_dominant_cycles_df.iterrows():

                peak_period = float(row['peak_periods'])
                start_rebuilt_signal_index = int(row['start_rebuilt_signal_index'])
                comparision_length = int(2.5 * peak_period) # comparision length
                length = int(len_series - start_rebuilt_signal_index)
                start_comparison_index = len_series - comparision_length
                
                # Sweep candidate amplitudes in descending order.
                for temp_amp in np.arange(self.MultiAn_detrended_max, 0, -0.01):                    
                    
                    
                    temp_circle_signal = pd.Series([0.0] * len_series)
                    time = np.linspace(0, length, length, endpoint=False)

                    temp_circle_signal[start_rebuilt_signal_index:] = temp_amp * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])

                    temp_rebuilt_signal = composite_dominant_cycle_signal + temp_circle_signal
                    
                    error = mean_squared_error(self.MultiAn_reference_detrended_data[start_comparison_index:], temp_rebuilt_signal[start_comparison_index:]) 
                    
                   
                    if(error < last_error):                        
                        best_amplitude = temp_amp 
                        best_error = error

                    last_error = error
                    
                    
                # Store the best amplitude found for this cycle.
                amplitudes.append(best_amplitude)
                best_fitness_value = best_error
                self.log_info(
                    "Mono-frequency best amplitude found",
                    function="multiperiod_analysis",
                    peak_period=row['peak_periods'],
                    best_amplitude=best_amplitude,
                    best_fitness_value=best_fitness_value,
                )
                
                # Add this cycle to the cumulative rebuilt signal.
                composite_dominant_cycle_signal = composite_dominant_cycle_signal + temp_rebuilt_signal
           
            self.log_debug(
                "Single-cycle best fitting amplitudes",
                function="multiperiod_analysis",
                amplitudes=amplitudes,
            )
            
        # -------------------------------------------------------
        # DEAP optimizer for amplitudes and optional frequency/phase tuning
        # -------------------------------------------------------
        
        elif(opt_algo_type == 'genetic_omny_frequencies'):

            
            self.debug_check_complex_values()

            
 
            self.log_timing('\n6. Genetics algo for best amplitudes identification, start inizializing', function="multiperiod_analysis")
    

            # Register DEAP fitness type once per process.
            if 'FitnessMulti' not in creator.__dict__:
                creator.create("FitnessMulti", base.Fitness, weights=(weigth,)) # loss are always negative so they must be maximized

            # Register DEAP individual type once per process.
            if 'Individual' not in creator.__dict__:
                creator.create("Individual", list, fitness=creator.FitnessMulti)

            # Configure individual creation, evaluation, crossover and mutation.
            toolbox = base.Toolbox()
            toolbox.register("individual", tools.initIterate, creator.Individual, self.MultiAn_initializeIndividual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.MultiAn_evaluateFitness)
            toolbox.register("mate", self.custom_crossover) # tools.cxTwoPoint)
            
            amp_low  = low_series.tolist()
            amp_up   = up_series.tolist()
            
            freq_low, freq_up = None, None
            phase_low, phase_up = None, None
            
            if self.frequencies_ft:
                freq_low = (self.MultiAn_dominant_cycles_df['peak_frequencies'] * 0.90).tolist()
                freq_up  = (self.MultiAn_dominant_cycles_df['peak_frequencies'] * 1.10).tolist()

            if self.phases_ft:
                two_pi = 2 * np.pi
                phase_low = (self.MultiAn_dominant_cycles_df['peak_phases'] - 0.10 * two_pi).tolist()
                phase_up  = (self.MultiAn_dominant_cycles_df['peak_phases'] + 0.10 * two_pi).tolist()

            
            def safe_segmented_mutation(individual, amp_low, amp_up, freq_low=None, freq_up=None, phase_low=None, phase_up=None, indpb=0.2):
                n = len(amp_low)
                levels = self.discretization_steps # 400
                cursor = 0
            
                for i in range(n):
                    if random.random() < indpb:
                        step = (amp_up[i] - amp_low[i]) / (levels - 1)
                        individual[i] = amp_low[i] + step * random.randint(0, levels - 1)
                cursor += n
            
                if freq_low is not None and freq_up is not None:
                    for i in range(n):
                        if random.random() < indpb:
                            step = (freq_up[i] - freq_low[i]) / (levels - 1)
                            individual[cursor + i] = freq_low[i] + step * random.randint(0, levels - 1)
                    cursor += n
            
                if phase_low is not None and phase_up is not None:
                    for i in range(n):
                        if random.random() < indpb:
                            step = (phase_up[i] - phase_low[i]) / (levels - 1)
                            individual[cursor + i] = phase_low[i] + step * random.randint(0, levels - 1)
            
                return (individual,)

            toolbox.register(
                "mutate",
                safe_segmented_mutation,
                amp_low=amp_low,
                amp_up=amp_up,
                freq_low=freq_low,
                freq_up=freq_up,
                phase_low=phase_low,
                phase_up=phase_up,
                indpb=0.2
            )


            
            toolbox.register("select", tools.selTournament, tournsize=3)

            if( enabled_multiprocessing == True):
                
                # Use spawn on Windows before registering pool.map.
                if multiprocessing.get_start_method() != 'spawn':
                    multiprocessing.set_start_method('spawn')

                cpu_count = multiprocessing.cpu_count()
                self.log_debug("Multiprocessing pool configured", function="multiperiod_analysis", cpu_count=cpu_count)

                pool = multiprocessing.Pool()
                toolbox.register("map", pool.map)


            # Create the initial population.
            population = toolbox.population(n=population_n)
            
            self.log_timing('6. Genetics algo for best amplitudes identification, end inizialization', function="multiperiod_analysis")
        

            for gen in range(NGEN):
                

                offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
                fits = toolbox.map(toolbox.evaluate, offspring)
                count = 0
                for fit, ind in zip(fits, offspring):                   
                    ind.fitness.values = fit
                    count += 1
                    

                population = toolbox.select(offspring, k=len(population))

                best_individual = tools.selBest(population, k=1)[0]
                best_fitness = best_individual.fitness.values


            if(enabled_multiprocessing == True):
                pool.close()
                pool.join()

            best_individual = tools.selBest(population, k=1)[0]
            best_fitness = best_individual.fitness.values


            self.log_debug(
                "DEAP optimization completed",
                function="multiperiod_analysis",
                best_individual=list(best_individual),
                best_fitness=best_fitness,
            )
                


            amplitudes, (frequencies, phases) = self.decode_individual(best_individual)
            self.MultiAn_dominant_cycles_df['best_amplitudes'] = amplitudes
            
            if frequencies is not None and len(frequencies) == len(self.MultiAn_dominant_cycles_df):
                self.MultiAn_dominant_cycles_df['frequency'] = frequencies
                self.MultiAn_dominant_cycles_df['best_frequencies'] = frequencies

            if phases is not None and len(phases) == len(self.MultiAn_dominant_cycles_df):
                self.MultiAn_dominant_cycles_df['phase'] = phases
                self.MultiAn_dominant_cycles_df['best_phases'] = phases


            best_fitness_value = best_fitness[0]
            self.MultiAn_dominant_cycles_df['best_fitness'] = best_fitness_value

            if self.is_log_enabled("DEBUG"):
                display(self.MultiAn_dominant_cycles_df)



        # -----------------------------------------------------------------------------------
        # C++ genetic optimizer for amplitudes and optional frequency/phase tuning
        # -----------------------------------------------------------------------------------

        elif opt_algo_type == 'cpp_genetic_amp_freq_phase':

            self.log_debug(
                "C++ optimization input columns",
                function="multiperiod_analysis",
                columns=list(self.MultiAn_dominant_cycles_df.columns),
            )
            
            self.log_timing('\n6. C++ Genetic Optimization start', function="multiperiod_analysis")
        
            n_cycles = len(self.MultiAn_dominant_cycles_df)
            
            if 'single_range_goertzel_peak_amplitudes' in self.MultiAn_dominant_cycles_df.columns:
                scaler = MinMaxScaler(feature_range=(0, detrended_abs_max))
                amp_init = scaler.fit_transform(self.MultiAn_dominant_cycles_df['single_range_goertzel_peak_amplitudes'].to_numpy().reshape(-1, 1)).flatten()

            else:
                amp_init = np.array([0.1] * n_cycles)

            freq_init = self.MultiAn_dominant_cycles_df['peak_frequencies'].to_numpy()
            
            phase_init = self.MultiAn_dominant_cycles_df['peak_phases'].to_numpy()

            self.log_debug("C++ amplitude initialization", function="multiperiod_analysis", amp_init=amp_init)


            initial_vector = []
            
            # Amplitudes.
            if amplitudes_inizialization_type == "random":
                initial_random_amplitudes = True
                initial_vector += amp_init.tolist()  # placeholder, saranno ignorate nel cpp
                
            elif amplitudes_inizialization_type == "all_equal_middle_value":
                initial_random_amplitudes = False
                middle_value = (self.MultiAn_detrended_max - self.MultiAn_detrended_min) / 2
                initial_vector += [middle_value] * n_cycles

                self.log_debug("Using middle amplitude initialization", function="multiperiod_analysis", middle_value=middle_value)
                
            elif amplitudes_inizialization_type == "transform_amplitudes":
                initial_random_amplitudes = False
                initial_vector += amp_init.tolist()
                
            else:
                raise ValueError(f"Invalid amplitudes_inizialization_type: {amplitudes_inizialization_type}")
            
            # Frequencies.
            initial_vector += freq_init.tolist()
            
            # Phases.
            initial_vector += phase_init.tolist()

                
            gene_length = len(initial_vector)
        
            # Build optimizer bounds in the same order as initial_vector.
            detrended_abs_max = abs(self.MultiAn_detrended_max - self.MultiAn_detrended_min)
            amp_min = [0.0] * n_cycles
            amp_max = [detrended_abs_max] * n_cycles

            self.log_debug("Reference detrended absolute max", function="multiperiod_analysis", detrended_abs_max=detrended_abs_max)
        
            if self.frequencies_ft:
                freq_min = (freq_init * 0.90).tolist()
                freq_max = (freq_init * 1.10).tolist()
            else:
                freq_min = freq_init.tolist()
                freq_max = freq_init.tolist()
            
            two_pi = 2 * np.pi
            if self.phases_ft:
                phase_min = (phase_init - 0.10 * two_pi).tolist()
                phase_max = (phase_init + 0.10 * two_pi).tolist()
            else:
                phase_min = phase_init.tolist()
                phase_max = phase_init.tolist()

        
            lb = amp_min.copy() + freq_min + phase_min
            ub = amp_max.copy() + freq_max + phase_max

            def active_vector_from_full(flat_list):
                full = np.asarray(flat_list, dtype=np.float64)
                active = [full[0:n_cycles]]
                if self.frequencies_ft:
                    active.append(full[n_cycles:2 * n_cycles])
                if self.phases_ft:
                    active.append(full[2 * n_cycles:3 * n_cycles])
                return np.concatenate(active)

            def fitness_func_cpp(flat_list):
                fitness_result = self.MultiAn_evaluateFitness(active_vector_from_full(flat_list), False)
                return float(fitness_result[0]) if isinstance(fitness_result, tuple) else float(fitness_result)


            if(enabled_multiprocessing):

                self.log_info("Running C++ multicore genetic optimizer", function="multiperiod_analysis")


                if self.period_related_rebuild_range:
                    # Compare only the recent segment implied by the longest cycle.
                    min_freq = self.MultiAn_dominant_cycles_df["peak_frequencies"].min()
                    peak_period = 1.0 / min_freq
                    period_related_rebuild_index = len(self.MultiAn_reference_detrended_data) - int(peak_period * self.period_related_rebuild_multiplier)
                    start_rebuild_index = max(
                        period_related_rebuild_index,
                        self.MultiAn_dominant_cycles_df["start_rebuilt_signal_index"].max()
                    )
                else:
                    if self.best_fit_start_back_period is None or self.best_fit_start_back_period == 0:
                        start_rebuild_index = self.MultiAn_dominant_cycles_df["start_rebuilt_signal_index"].max()
                    else:
                        start_rebuild_index = len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period


                best_flat = run_genetic_algorithm_multicore(
                    self.MultiAn_reference_detrended_data.tolist(),  # reference_signal
                    population_n,
                    CXPB,
                    MUTPB,
                    NGEN,
                    gene_length,
                    lb,
                    ub,
                    discretization_steps,
                    initial_vector,
                    n_cycles,
                    genetic_elitism=True,
                    elitism_elements=10,
                    initial_random_amplitudes=initial_random_amplitudes,
                    optimize_amplitudes=True,               # sempre True
                    optimize_frequencies=self.frequencies_ft,
                    optimize_phases=self.phases_ft,
                    start_rebuild_index=int(start_rebuild_index),
                    period_related_rebuild_range=self.period_related_rebuild_range,
                    period_multiplier=self.period_related_rebuild_multiplier,
                    peak_frequencies=self.MultiAn_dominant_cycles_df["peak_frequencies"].astype(float).tolist(),
                    peak_phases=self.MultiAn_dominant_cycles_df["peak_phases"].astype(float).tolist(),
                    peak_periods=self.MultiAn_dominant_cycles_df["peak_periods"].astype(float).tolist(),
                    start_indices=self.MultiAn_dominant_cycles_df["start_rebuilt_signal_index"].astype(int).tolist(),
                    best_fit_start_back_period=int(self.best_fit_start_back_period or 0),
                    fitness_type=str(self.MultiAn_fitness_type),
                    seed=int(random_seed) if random_seed is not None else -1,
                    threads=multiprocessing.cpu_count()
                )




            else:

                self.log_info("Running C++ single-core genetic optimizer", function="multiperiod_analysis")

                best_flat = run_genetic_algorithm(
                    fitness_func_cpp,
                    population_n,
                    CXPB,
                    MUTPB,
                    NGEN,
                    gene_length,
                    lb,
                    ub,
                    discretization_steps,
                    initial_vector,
                    n_cycles,
                    genetic_elitism=True,
                    elitism_elements=10,
                    initial_random_amplitudes=initial_random_amplitudes,
                    optimize_amplitudes=True,
                    optimize_frequencies=self.frequencies_ft,
                    optimize_phases=self.phases_ft,
                    seed=int(random_seed) if random_seed is not None else -1,
                )

            
            n = len(self.MultiAn_dominant_cycles_df)
            if len(best_flat) != 3 * n:
                raise ValueError(f"best_flat ha {len(best_flat)} elementi, ma me ne aspettavo {3 * n}")

            
            amp = best_flat[0:n]
            freq = best_flat[n:2*n]
            phase = best_flat[2*n:3*n]
            
            self.MultiAn_dominant_cycles_df["best_amplitudes"] = amp
            self.MultiAn_dominant_cycles_df["best_frequencies"] = freq
            assert len(phase) == len(self.MultiAn_dominant_cycles_df), f"phase ha {len(phase)} elementi, ma me ne aspetto {len(self.MultiAn_dominant_cycles_df)}"
            self.MultiAn_dominant_cycles_df["best_phases"] = phase


            amplitudes = amp




            individual = active_vector_from_full(best_flat)
            self.MultiAn_dominant_cycles_df['best_fitness'] = self.MultiAn_evaluateFitness(individual, False)


        
            self.log_debug(
                "C++ genetic optimization completed",
                function="multiperiod_analysis",
                best_fitness=self.MultiAn_dominant_cycles_df["best_fitness"].iloc[0],
            )
        
            self.log_timing('6. C++ Genetic Optimization end', function="multiperiod_analysis")
        

            best_fitness_value = self.MultiAn_dominant_cycles_df['best_fitness']


        # -----------------------------------------------------------------------------------
        # NLopt optimizer for amplitudes and optional frequency/phase tuning
        # -----------------------------------------------------------------------------------

        elif(opt_algo_type == 'nlopt_amplitudes_freqs_phases'):
            
            self.log_timing('\n6. NLopt optimization: start', function="multiperiod_analysis")
            best_individual, best_fitness = self.MultiAn_optimize_NLOPT(
                                                                optimize_frequencies = self.frequencies_ft,
                                                                optimize_phases = self.phases_ft,
                                                                freq_range_pct = 0.10,
                                                                phase_range_pct = 0.10,
                                                                maxeval = NGEN
                                            )

            best_fitness_value = best_fitness

            amplitudes, (frequencies, phases) = self.decode_individual(best_individual)
            self.MultiAn_dominant_cycles_df["amplitude"] = amplitudes
            if frequencies is not None:
                self.MultiAn_dominant_cycles_df["frequency"] = frequencies
            if phases is not None:
                self.MultiAn_dominant_cycles_df["phase"] = phases


            


            self.log_info("NLopt optimization completed", function="multiperiod_analysis", best_fitness=best_fitness_value)


            self.log_timing('6. NLopt optimization: end', function="multiperiod_analysis")


        # -------------------------------------------------------------------------------------------------------------------------
        # Hyperopt TPE/ATPE optimizer for amplitudes and optional frequency/phase tuning
        # -------------------------------------------------------------------------------------------------------------------------

        elif(opt_algo_type ==  'tpe' or opt_algo_type ==  'atpe'):

            from hyperopt import STATUS_OK  # puo anche stare fuori dalla funzione una volta sola

            # Hyperopt minimizes loss, so the objective returns the fitness loss.
            def objective(params):
                individual = []
                n_cycles = len(self.MultiAn_dominant_cycles_df)
            
                # Amplitudes are always part of the optimization vector.
                amplitude_values = [params[f'amplitude_{i}'] for i in range(n_cycles)]
                individual.extend(amplitude_values)
            
                # Frequencies are included only when enabled for this run.
                if self.frequencies_ft:
                    frequency_values = [params[f'frequency_{i}'] for i in range(n_cycles)]
                    individual.extend(frequency_values)
                    self.MultiAn_dominant_cycles_df['frequency'] = frequency_values
                else:
                    self.MultiAn_dominant_cycles_df['frequency'] = self.MultiAn_dominant_cycles_df['peak_frequencies']
            
                # Phases are included only when enabled for this run.
                if self.phases_ft:
                    phase_values = [params[f'phase_{i}'] for i in range(n_cycles)]
                    individual.extend(phase_values)
                    self.MultiAn_dominant_cycles_df['phase'] = phase_values
                else:
                    self.MultiAn_dominant_cycles_df['phase'] = self.MultiAn_dominant_cycles_df['peak_phases']
            
                

                # Keep return_list_type=True because MultiAn_evaluateFitness returns a tuple.
                fitness = self.MultiAn_evaluateFitness(individual, return_list_type=True)
            
                # Hyperopt requires a dict with loss and status.
                return {"loss": float(fitness[0]), "status": STATUS_OK}


            # Define the Hyperopt search space in vector order.
            space = {f'amplitude_{i}': hp.uniform(f'amplitude_{i}', low_series[i], up_series[i]) for i in range(len(low_series))}

            n_cycles = len(self.MultiAn_dominant_cycles_df)
            
            if self.frequencies_ft:
                base_frequencies = self.MultiAn_dominant_cycles_df.get("frequency", self.MultiAn_dominant_cycles_df["peak_frequencies"]).values
                freq_low = base_frequencies * (1 - 0.10)
                freq_up = base_frequencies * (1 + 0.10)
                space.update({f'frequency_{i}': hp.uniform(f'frequency_{i}', freq_low[i], freq_up[i]) for i in range(len(freq_low))})
            
            if self.phases_ft:
                base_phases = self.MultiAn_dominant_cycles_df.get("phase", self.MultiAn_dominant_cycles_df["peak_phases"]).values
                phase_low = base_phases - (2 * np.pi * 0.10)
                phase_up = base_phases + (2 * np.pi * 0.10)
                space.update({f'phase_{i}': hp.uniform(f'phase_{i}', phase_low[i], phase_up[i]) for i in range(len(phase_low))})


            # Run the selected Hyperopt algorithm.
            if(opt_algo_type ==  'tpe'):
                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=NGEN)

            if(opt_algo_type ==  'atpe'):
                best = fmin(fn=objective, space=space, algo=atpe.suggest, max_evals=NGEN)



            # Copy the best parameter values back into the dominant-cycle table.
            n = len(self.MultiAn_dominant_cycles_df)
            amplitudes = [best[f'amplitude_{i}'] for i in range(n)]
            self.MultiAn_dominant_cycles_df['best_amplitudes'] = amplitudes
            best_amplitudes = amplitudes

            self.log_debug("Hyperopt best amplitudes", function="multiperiod_analysis", best_amplitudes=best_amplitudes)
            
            if self.frequencies_ft:
                frequencies = [best[f'frequency_{i}'] for i in range(n)]
                self.MultiAn_dominant_cycles_df['frequency'] = frequencies
                self.MultiAn_dominant_cycles_df['best_frequencies'] = frequencies
                self.log_debug("Hyperopt best frequencies", function="multiperiod_analysis", best_frequencies=frequencies)
            
            if self.phases_ft:
                phases = [best[f'phase_{i}'] for i in range(n)]
                self.MultiAn_dominant_cycles_df['phase'] = phases
                self.MultiAn_dominant_cycles_df['best_phases'] = phases
                self.log_debug("Hyperopt best phases", function="multiperiod_analysis", best_phases=phases)



            # Re-evaluate the best point and store the resulting scalar loss.
            best_evaluation = objective(best)
            best_fitness_value = float(best_evaluation["loss"])
            self.MultiAn_dominant_cycles_df['best_fitness'] = best_fitness_value

            if self.is_log_enabled("DEBUG"):
                display(self.MultiAn_dominant_cycles_df)

            amplitudes = best_amplitudes

        else:

            self.log_error(
                "Optimization type not supported",
                function="multiperiod_analysis",
                opt_algo_type=opt_algo_type,
            )
            return None, None, None, None, None, None


        # -----------------------------------------------------------------------
        # Create composite signals from optimized cycle parameters
        # -----------------------------------------------------------------------

        if self.is_log_enabled("DEBUG"):
            self.log_timing('\n9. Genetics start composite signal creation', function="multiperiod_analysis")


        temp_circle_signal = []
        composite_dominant_cycle_signal = []

        # Projection length follows the longest generated single-range output.
        len_series = len(self.data) #len(self.MultiAn_reference_detrended_data)

        # Use the longest single-range index as the composite signal index.
        self.log_debug(
            "Composite signal sizing",
            function="multiperiod_analysis",
            len_series=len_series,
            elaborated_data_series_count=len(elaborated_data_series),
            max_length_series_index=max_length_series_index,
        )

        
        df_indexes_list = elaborated_data_series[max_length_series_index].index

        self.log_timing('\t9.a. Genetics start composite_signal', function="multiperiod_analysis")
        composite_signal = self.cicles_composite_signals(max_length_series, amplitudes,self.MultiAn_dominant_cycles_df, df_indexes_list, 'composite_signal')       

        self.log_timing('\t9.b. Genetics: end composite_signal, start alignmentsKPI', function="multiperiod_analysis")
        
        alignmentsKPI = pd.Series()
        weigthed_alignmentsKPI = pd.Series()
        
        if(enable_cycles_alignment_analysis == True):
                                        
            alignmentsKPI, weigthed_alignmentsKPI = self.MultiAn_cyclesAlignKPI(composite_signal.drop(['composite_signal'], axis=1),
                                                                            max_start_rebuilt_signal_index,
                                                                            amplitudes,
                                                                            self.MultiAn_dominant_cycles_df['peak_periods'])
            
        else:
            self.log_debug("Alignment KPI analysis disabled", function="multiperiod_analysis")

        if self.is_log_enabled("DEBUG"):
            self.log_timing('\t9.c. Genetics: end alignmentsKPI, start cicles_composite_signals', function="multiperiod_analysis")

        temp = self.cicles_composite_signals(max_length_series, goertzel_amplitudes, goertzel_best_refactoring_df, df_indexes_list, 'goertzel_composite_signal')
        
        composite_signal = pd.concat([composite_signal, temp], axis=1)

        if self.is_log_enabled("DEBUG"):
            self.log_timing('10. Genetics end composite and alignmentsKPI signal creation', function="multiperiod_analysis")

        

        start_index = index_of_max_time_for_cd - num_samples    


        # -------------------------------------------------
        # Scale composite and alignment signals for output
        # -------------------------------------------------
        
        self.log_debug("Scaling output signals", function="multiperiod_analysis")
        self.log_timing('\n11. Genetics end composite signal creation', function="multiperiod_analysis")

        scaled_signals = pd.DataFrame()

        CDC_min = elaborated_data_series[index_detrended_data]['detrended'][start_index:].min(skipna=True)
        CDC_max = elaborated_data_series[index_detrended_data]['detrended'][start_index:].max(skipna=True)


        CDC_scaler = MinMaxScaler(feature_range=(CDC_min , CDC_max ))

        
        scaled_signals['scaled_composite_signal'] = scaled_composite_signal  = CDC_scaler.fit_transform( composite_signal['composite_signal'].values.reshape(-1, 1) ).flatten()
        scaled_signals['scaled_goertzel_composite_signal'] = scaled_goertzel_composite_signal = CDC_scaler.fit_transform( composite_signal['goertzel_composite_signal'].values.reshape(-1, 1) ).flatten()
        
        # Align detrended values to the projected scaled-signal length.
        detrended_values = elaborated_data_series[index_detrended_data]['detrended'].values
        scaled_signals_len = len(scaled_signals['scaled_composite_signal'])  # Assumiamo che sia questa la lunghezza target

        # If the detrended reference is shorter, left-pad it with NaN.
        if len(detrended_values) < scaled_signals_len:
            num_nans_to_add = scaled_signals_len - len(detrended_values)
            detrended_values = np.concatenate([np.full(num_nans_to_add, np.nan), detrended_values])

        # Store the aligned detrended reference.
        scaled_signals['scaled_detrended'] = scaled_detrended  = detrended_values

        
        if(enable_cycles_alignment_analysis == True):

            scaled_signals['scaled_alignmentsKPI'] = scaled_alignmentsKPI = scaler.fit_transform( alignmentsKPI.values.reshape(-1, 1)).flatten()
            scaled_signals['scaled_weigthed_alignmentsKPI'] = scaled_weigthed_alignmentsKPI = scaler.fit_transform( weigthed_alignmentsKPI.values.reshape(-1, 1)).flatten()
            
        else:
            scaled_signals['scaled_alignmentsKPI'] = scaled_alignmentsKPI = np.zeros(len(scaled_signals['scaled_composite_signal']))
            scaled_signals['scaled_weigthed_alignmentsKPI'] = scaled_weigthed_alignmentsKPI = np.zeros(len(scaled_signals['scaled_composite_signal']))

        if self.is_log_enabled("DEBUG"):
            self.log_timing('12. Genetics end composite signal creation', function="multiperiod_analysis")
            
   

        self.log_debug(
            "Composite signal ready before chart plotting",
            function="multiperiod_analysis",
            index_of_max_time_for_cd=index_of_max_time_for_cd,
        )
        # -------------------------------------------------
        # Optional Plotly diagnostics
        # -------------------------------------------------

        if self.is_log_enabled("DEBUG"):
            self.log_debug("Before charts plot", function="multiperiod_analysis")
            

        if(show_charts == True):

            self.plot_multiperiod_analysis_charts(
                reduced_data=reduced_data,
                data_column_name=data_column_name,
                composite_signal=composite_signal,
                elaborated_data_series=elaborated_data_series,
                max_length_series_index=max_length_series_index,
                scaled_composite_signal=scaled_composite_signal,
                scaled_goertzel_composite_signal=scaled_goertzel_composite_signal,
                scaled_detrended=scaled_detrended,
                scaled_alignmentsKPI=scaled_alignmentsKPI,
                scaled_weigthed_alignmentsKPI=scaled_weigthed_alignmentsKPI,
                index_of_max_time_for_cd=index_of_max_time_for_cd,
            )


        return elaborated_data_series, signals_results_series, composite_signal, configurations_series, None, None, index_of_max_time_for_cd, scaled_signals, best_fitness_value
