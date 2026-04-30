"""Optimization helpers for the legacy cyPredict workflows."""

from datetime import datetime
from decimal import Decimal
import multiprocessing
import random
import traceback

from deap import algorithms, base, creator, tools
from IPython.display import clear_output, display
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error


class OptimizationMixin:
    """Optimization utilities shared by genetic and multi-period workflows."""

    def custom_crossover(self, ind1, ind2):
        """Apply the legacy DEAP crossover strategy.

        Parameters
        ----------
        ind1, ind2 : deap creator.Individual or sequence-like
            Individuals to cross over. When both contain more than one gene the
            historical two-point crossover is used. Single-gene individuals use
            uniform crossover with probability ``0.5``.

        Returns
        -------
        tuple
            The pair returned by the selected DEAP crossover operator.
        """
        from deap import tools

        if len(ind1) > 1 and len(ind2) > 1:
            return tools.cxTwoPoint(ind1, ind2)

        return tools.cxUniform(ind1, ind2, 0.5)

    # Individual random definition function
    def genOpt_initializeIndividual(self):
        
        if(self.detrend_type == 'hp_filter'):
            
            if(self.period_related_rebuild_range == True):  
                
                return [random.randint(self.genOpt_num_samples_min, self.genOpt_num_samples_max),
                        random.randint(self.genOpt_final_kept_n_dominant_circles_min, self.genOpt_final_kept_n_dominant_circles_max),
                        random.randint(self.genOpt_min_period_min, self.genOpt_min_period_max),
                        random.randint(self.genOpt_max_period_min, self.genOpt_max_period_max),
                        random.choice( self.genOpt_logarithmic_sequence ),
                        random.choice(self.period_related_rebuild_multiplier_sequence)
                       ]            

            else: 
                
                return [random.randint(self.genOpt_num_samples_min, self.genOpt_num_samples_max),
                        random.randint(self.genOpt_final_kept_n_dominant_circles_min, self.genOpt_final_kept_n_dominant_circles_max),
                        random.randint(self.genOpt_min_period_min, self.genOpt_min_period_max),
                        random.randint(self.genOpt_max_period_min, self.genOpt_max_period_max),
                        random.choice( self.genOpt_logarithmic_sequence )
                       ]
        
        
        elif(self.detrend_type == 'linear'):                
                
            if(self.period_related_rebuild_range == True):                           
            
                return [
                        random.randint(self.genOpt_num_samples_min, self.genOpt_num_samples_max),
                        random.randint(self.genOpt_final_kept_n_dominant_circles_min, self.genOpt_final_kept_n_dominant_circles_max),
                        random.randint(self.genOpt_min_period_min, self.genOpt_min_period_max),
                        random.randint(self.genOpt_max_period_min, self.genOpt_max_period_max),                        
                        random.choice(self.linear_filter_window_size_multiplier_sequence),
                        random.choice(self.period_related_rebuild_multiplier_sequence)
                       ]  
            else:
                
                return [
                        random.randint(self.genOpt_num_samples_min, self.genOpt_num_samples_max),
                        random.randint(self.genOpt_final_kept_n_dominant_circles_min, self.genOpt_final_kept_n_dominant_circles_max),
                        random.randint(self.genOpt_min_period_min, self.genOpt_min_period_max),
                        random.randint(self.genOpt_max_period_min, self.genOpt_max_period_max),
                        random.choice(self.linear_filter_window_size_multiplier_sequence)
                       ]  
                
        
        




            



    def discretized_uniform(self, low, up, levels=400):
        step = (up - low) / (levels - 1)
        return low + step * random.randint(0, levels - 1)



    def MultiAn_initializeIndividual(self):
        n = len(self.MultiAn_dominant_cycles_df)
        individual = []
    
        # Initialize amplitudes on the configured discrete grid.
        for _ in range(n):
            individual.append(self.discretized_uniform(0.0, self.MultiAn_detrended_max))
    
        if self.frequencies_ft:
            base_freqs = self.MultiAn_dominant_cycles_df['peak_frequencies'].values
            for i in range(n):
                f_low = base_freqs[i] * 0.90
                f_up  = base_freqs[i] * 1.10
                individual.append(self.discretized_uniform(f_low, f_up))
    
        if self.phases_ft:
            base_phases = self.MultiAn_dominant_cycles_df['peak_phases'].values
            two_pi = 2 * np.pi
            for i in range(n):
                p_low = base_phases[i] - 0.10 * two_pi
                p_up  = base_phases[i] + 0.10 * two_pi
                individual.append(self.discretized_uniform(p_low, p_up))
    
        return individual



    
    def MultiAn_evaluateFitness(self, individual, return_list_type=True):
        """
        Se 'individual' contiene ampiezze (ed eventualmente frequenze e fasi),
        costruiamo l'array NumPy e lo passiamo a C. 
        La parte C si aspetta un array di lunghezza n, 
        oppure 2n (se frequencies_ft=True) oppure 3n (se phases_ft=True),
        in quest'ordine: [amp_1, amp_2, ..., amp_n, freq_1, ..., freq_n, phase_1, ..., phase_n].
        """
    
        import numpy as np
        from cyfitness import evaluate_fitness
    
        # 1) Calcoliamo quanti cicli (n) abbiamo
        n = len(self.MultiAn_dominant_cycles_df)
        if n == 0:
            # Nessun ciclo => fitness altissimo
            return (1e9,) if return_list_type else 1e9
    
        # 2) Determiniamo la lunghezza attesa di 'individual'
        freq_n = n if self.frequencies_ft else 0
        phase_n = n if self.phases_ft else 0
        expected_size = n + freq_n + phase_n
    
        # 3) Controllo di coerenza
        if len(individual) != expected_size:
            raise ValueError(
                f"Dimensione di 'individual' incoerente: "
                f"attesi {expected_size} valori, ne ho {len(individual)}"
            )
    
        # 4) Convertiamo 'individual' in un array NumPy float64
        individual_array = np.array(individual, dtype=np.float64)
    
        # 5) Preparo gli altri parametri così come richiesti dalla firma "OOOppiiidsi"
        frequencies_ft_int = int(self.frequencies_ft)
        phases_ft_int = int(self.phases_ft)
        len_series = len(self.MultiAn_reference_detrended_data)
        best_fit_start_back_period_int = self.best_fit_start_back_period or 0
        period_related_rebuild_range_int = int(self.period_related_rebuild_range)
        period_related_rebuild_multiplier_float = float(self.period_related_rebuild_multiplier)
        fitness_type_str = str(self.MultiAn_fitness_type)
        return_list_type_int = int(return_list_type)
    
        # Il DataFrame conterrà campi minimi come 'start_rebuilt_signal_index' e 'peak_periods'
        # (oppure 'peak_frequencies' se preferisci).
        # Non serve aggiungere 'amplitude'/'frequency'/'phase' nel dict, 
        # perché stiamo passando quei parametri in 'individual_array'.
        cycles_dict = self.MultiAn_dominant_cycles_df.to_dict(orient="records")
    
        # 6) Chiamiamo evaluate_fitness di C++ con l'array che abbiamo costruito
        fitness_result = evaluate_fitness(
            individual_array,                         # <--- array con ampiezze + freq + fasi
            self.MultiAn_reference_detrended_data,    # reference_data
            cycles_dict,                              # list di dict (solo info di contesto)
            frequencies_ft_int,                       # se in 'individual_array' ci sono freq
            phases_ft_int,                            # se in 'individual_array' ci sono fasi
            len_series,
            best_fit_start_back_period_int,
            period_related_rebuild_range_int,
            period_related_rebuild_multiplier_float,
            fitness_type_str,
            return_list_type_int
        )
    
        # 7) Se return_list_type==1, C++ ci restituirà (fitness,),
        #    altrimenti un float (tipo 123.45).
        return fitness_result
    




    def decode_individual(self, individual):
        n = len(self.MultiAn_dominant_cycles_df)
        cursor = 0
        
        amplitudes = individual[cursor:cursor + n]
        cursor += n
        
        frequencies = None
        if self.frequencies_ft:
            frequencies = individual[cursor:cursor + n]
            cursor += n
        
        phases = None
        if self.phases_ft:
            phases = individual[cursor:cursor + n]
            cursor += n
        
        return np.array(amplitudes), (
            np.array(frequencies) if frequencies is not None else None,
            np.array(phases) if phases is not None else None
        )



    
    def MultiAn_optimize_NLOPT(self,
                               optimize_frequencies=False,
                               optimize_phases=False,
                               freq_range_pct=0.10,
                               phase_range_pct=0.10,
                               maxeval=10000):
    
        import nlopt
        import numpy as np
    
        df = self.MultiAn_dominant_cycles_df
        n_cycles = len(df)
    
        x0 = []
        lb = []
        ub = []
    
        # Amplitude bounds.
        for i in range(n_cycles):
            amp_min = 0
            amp_max = self.MultiAn_detrended_max
            x0.append((amp_min + amp_max) / 2)
            lb.append(amp_min)
            ub.append(amp_max)
    
        # Optional frequency bounds around Goertzel estimates.
        if optimize_frequencies:
            for i in range(n_cycles):
                base_freq = df['peak_frequencies'].iloc[i]
                lo = base_freq * (1 - freq_range_pct)
                hi = base_freq * (1 + freq_range_pct)
                x0.append((lo + hi) / 2)
                lb.append(lo)
                ub.append(hi)
    
        # Optional phase bounds around Goertzel estimates.
        if optimize_phases:
            for i in range(n_cycles):
                base_phase = df['peak_phases'].iloc[i]
                lo = base_phase - phase_range_pct * 2 * np.pi
                hi = base_phase + phase_range_pct * 2 * np.pi
                x0.append(base_phase)
                lb.append(lo)
                ub.append(hi)

    
        # Converti tutto a float puri per evitare errori
        x0 = [float(v) for v in x0]
        lb = [float(v) for v in lb]
        ub = [float(v) for v in ub]
        
        EPSILON = 1e-8
        
        # Correggi upper bounds se troppo stretti
        ub = [ub[i] if ub[i] > lb[i] + EPSILON else lb[i] + EPSILON for i in range(len(ub))]
        
        # Correggi x0 se tocca (o supera) ub
        for i in range(len(x0)):
            if x0[i] >= ub[i]:
                self.log_debug(
                    "Adjusted NLopt x0 touching upper bound",
                    function="MultiAn_optimize_NLOPT",
                    index=i,
                    original_value=x0[i],
                    upper_bound=ub[i],
                )
                x0[i] = ub[i] - EPSILON

        def loss(x, grad):
            result = self.MultiAn_evaluateFitness(x)
            # Se è una tupla con un solo elemento, estrai il valore
            if isinstance(result, tuple):
                result = result[0]
            return float(result)

        
        # ⚠️ Ora scegli l’algoritmo in base alla dimensionalità
        if len(x0) <= 8:
            algo = nlopt.LN_COBYLA
            self.log_info("Using NLopt algorithm", function="MultiAn_optimize_NLOPT", algorithm="COBYLA")
        else:
            algo = nlopt.GN_ISRES
            self.log_info("Using NLopt algorithm", function="MultiAn_optimize_NLOPT", algorithm="ISRES", reason="fallback for high dimension")
        
        # ⚠️ Ora istanzia opt
        opt = nlopt.opt(algo, len(x0))
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_min_objective(loss)
        opt.set_maxeval(maxeval)
        
        # Debug finale
        for i in range(len(x0)):
            assert isinstance(x0[i], float), f"x0[{i}] non è float puro!"
            assert lb[i] <= x0[i] <= ub[i], f"x0[{i}] fuori dai bounds!"
        
        best_x = opt.optimize(x0)
        best_fitness = opt.last_optimum_value()

        
        return best_x, best_fitness        


    
    def MultiAn_evaluateFitness_py(self, individual, return_list_type = True):

        scaler = self.scaler

        amplitudes = individual

        temp_circle_signal = []
        composite_dominant_cycle_signal = []
        len_series = len(self.MultiAn_reference_detrended_data)
        composite_dominant_cycle_signal = pd.Series([0.0] * len_series)
        
        # 1. each component is rebuilt starting from row['start_rebuilt_signal_index'] that is the index from which was
        #    started the Goertzel transform --> it must be rebuilt from here for not violating the coherence with the 
        #    trasnform given phase
        # 2. then if period_related_rebuild_range == False it is added entirerly in the final composite signal; otherwise it is added
        #    just the letst and most actual part of len_series - int((row['peak_periods'] * self.period_related_rebuild_multiplier))
        #    periods 
        # 3. finally it evaluate a period equal to the longest rebuilt cycle if best_fit_start_back_period == None or 0; otherwise, it 
        #    starts the comparison between the original detrended signal and the rebuilt one from the index 

        for index, row in self.MultiAn_dominant_cycles_df.iterrows():

            length = int(len_series - row['start_rebuilt_signal_index'])
            temp_circle_signal = pd.Series([0.0] * len_series)
            time = np.linspace(0, length, length, endpoint=False)

            temp_circle_signal[int(row['start_rebuilt_signal_index']):] = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
            
            # CUT IT KEEPING LOWER VALUES ON THE RIGHT IF self.period_related_rebuild_range == True
            if(self.period_related_rebuild_range == True):
                
                period_related_rebuild_index = len_series - int((row['peak_periods'] * self.period_related_rebuild_multiplier))
                
                if(period_related_rebuild_index < row['start_rebuilt_signal_index']):
                    period_related_rebuild_index = row['start_rebuilt_signal_index']
                    self.log_debug(
                        "Period-related rebuild index before cycle start",
                        function="MultiAn_evaluateFitness_py",
                        adjusted_index=period_related_rebuild_index,
                        start_rebuilt_signal_index=row['start_rebuilt_signal_index'],
                    )
                    
                temp_circle_signal_2 = [0.0] * len(temp_circle_signal)

                # Copy values from start_evaluation_index onwards
                temp_circle_signal_2[period_related_rebuild_index:] = temp_circle_signal[period_related_rebuild_index:]
                temp_circle_signal = temp_circle_signal_2

            composite_dominant_cycle_signal += temp_circle_signal
    
        if(self.best_fit_start_back_period is None or self.best_fit_start_back_period == 0):
            
            #  it takes the max index, meaning it evaluates the lowest period used for the Goertzel transform
            max_pos = self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].max()
            
        else:
            # it takes a given index
            max_pos = len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period
            
        
        if(self.MultiAn_fitness_type == "mse"):


            fitness = mean_squared_error(self.MultiAn_reference_detrended_data[max_pos:], composite_dominant_cycle_signal[max_pos:]) # 
            
        if(self.MultiAn_fitness_type == "just_mins_maxes"):

            mins = argrelextrema(composite_dominant_cycle_signal[max_pos:].values, np.less)[0]
            maxes = argrelextrema(composite_dominant_cycle_signal[max_pos:].values, np.greater)[0]
            peaks_indexes = np.concatenate([mins, maxes])


            fitness = mean_squared_error(self.MultiAn_reference_detrended_data[max_pos:].iloc[peaks_indexes], composite_dominant_cycle_signal[max_pos:].iloc[peaks_indexes])


        if(return_list_type == True):
            return (fitness, )

        else:
            return fitness



    # Fitness function
    def genOpt_evaluateMSEFitness(self, individual):

        data = self.data
        last_date = self.genOpt_last_date
        
        self.log_debug("Evaluating MSE fitness", function="genOpt_evaluateMSEFitness", last_date=self.genOpt_last_date)
        periods_number = self.genOpt_periods_number
        period_related_rebuild_multiplier = 1
        linear_filter_window_size_multiplier = 1
        hp_filter_lambda = 1

        if(self.detrend_type == 'hp_filter'):
            
            if(self.period_related_rebuild_range == True):                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda, period_related_rebuild_multiplier  = individual
                
                if((period_related_rebuild_multiplier < 2) or (period_related_rebuild_multiplier > 6)):
                    self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="period_related_rebuild_multiplier out of bounds")
                    return (1e9,)  # Vincolo violato
                
            else:                                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda  = individual            
            
            
            if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
                self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="hp_filter_lambda out of bounds")
                return (1e9,)  # Vincolo violato
            
            
            
        elif(self.detrend_type == 'linear'):
            
            if(self.period_related_rebuild_range == True):                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, linear_filter_window_size_multiplier, period_related_rebuild_multiplier = individual
                
                if((period_related_rebuild_multiplier < 1) or (period_related_rebuild_multiplier > 5)):
                    self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="period_related_rebuild_multiplier out of bounds")
                    return (1e9,)  # Vincolo violato
                
            else:                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, linear_filter_window_size_multiplier  = individual                
                
            if((linear_filter_window_size_multiplier < 0.5) or (linear_filter_window_size_multiplier > 2)):
                    self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="linear_filter_window_size_multiplier out of bounds")
                    return (1e9,)  # Vincolo violato

        # Constraints definition
        if num_samples < max_period*2:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="num_samples < max_period*2")
            return (1e9,)  # Vincolo violato
        if num_samples > 5000 or num_samples < 30:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="num_samples outside allowed range")
            return (1e9,)  # Vincolo violatoo
        if final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="final_kept_n_dominant_circles outside allowed range")
            return (1e9,)  # Vincolo violato
        if min_period < 1 or min_period > 1024:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="min_period outside allowed range")
            return (1e9,)  # Vincolo violato
        if max_period < 7 or max_period > 1024:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="max_period outside allowed range")
            return (1e9,)  # Vincolo violato
        if min_period + 2 > max_period:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="min_period + 2 > max_period")
            return (1e9,)  # Vincolo violato
        if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
            self.log_debug("Constraint violation", function="genOpt_evaluateMSEFitness", constraint="hp_filter_lambda out of bounds")
            return (1e9,)  # Vincolo violato


        # Use CDC_vs_detrended_correlation_sum for determining the values of the fitness function
        try:


            fitness = self.CDC_vs_detrended_correlation_sum(
                last_date=last_date,
                data=data,
                periods_number = periods_number,
                num_samples=num_samples,
                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                min_period=min_period,
                max_period=max_period,
                hp_filter_lambda=hp_filter_lambda,                                
                opt_algo_type = self.opt_algo_type, 
                detrend_type = self.detrend_type, #'linear', #'hp_filter',
                windowing = self.windowing,
                kaiser_beta = self.kaiser_beta,
                linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                period_related_rebuild_range = self.period_related_rebuild_range,
                period_related_rebuild_multiplier = period_related_rebuild_multiplier
                
            )
            


        except Exception as e:
            self.log_error(
                "Exception while evaluating MSE fitness",
                function="genOpt_evaluateMSEFitness",
                error=e,
                traceback=traceback.format_exc(),
                last_date=last_date,
                periods_number=periods_number,
                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                min_period=min_period,
                max_period=max_period,
                best_fit_start_back_period=self.best_fit_start_back_period,
                hp_filter_lambda=hp_filter_lambda,
                linear_filter_window_size_multiplier=linear_filter_window_size_multiplier,
                period_related_rebuild_multiplier=period_related_rebuild_multiplier,
            )

            return (1e9, )

        else:

            return fitness,



    # Fitness function
    def genOpt_evaluateFitness(self, individual):

        data = self.data
        last_date = self.genOpt_last_date
        logarithmic_sequence = self.genOpt_logarithmic_sequence
        periods_number = self.genOpt_periods_number

        num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda = individual


        # Constraints definition
        if num_samples < max_period*2:
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)  # Vincolo violato
        if num_samples > 1200 or num_samples < 30:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="num_samples outside allowed range")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="final_kept_n_dominant_circles outside allowed range")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if min_period < 1 or min_period > 200:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="min_period outside allowed range")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if max_period < 7 or max_period > 400:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="max_period outside allowed range")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if min_period + 2 > max_period:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="min_period + 2 > max_period")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
            self.log_debug("Constraint violation", function="genOpt_evaluateFitness", constraint="hp_filter_lambda out of bounds")
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato


        # Use trade_predicted_dominant_cicles_peaks_sum for determining the values of the fitness function
        try:
            value_sum, max_loss, max_cumulative_loss, profits_sum, profits_count, losses_sum, losses_count = self.trade_predicted_dominant_cicles_peaks_sum(
                data,
                last_date=last_date,                
                periods_number = periods_number,
                num_samples=num_samples,
                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                min_period=min_period,
                max_period=max_period,
                hp_filter_lambda=hp_filter_lambda
            )


        except Exception as e:

            self.log_error(
                "Exception while evaluating trading fitness",
                function="genOpt_evaluateFitness",
                error=e,
                traceback=traceback.format_exc(),
            )
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)

        else:            

            if(profits_count + losses_count > 0):
                count_profits_vs_losses = np.abs(profits_count / (profits_count + losses_count))
                profits_vs_losses = np.abs(profits_sum / (profits_sum + np.abs(losses_sum)))
            else:
                count_profits_vs_losses = 0
                profits_vs_losses = 0


            return (value_sum,
                    max_loss,
                    max_cumulative_loss,
                    profits_sum,
                    profits_count,
                    losses_sum,
                    losses_count,
                    count_profits_vs_losses,
                    profits_vs_losses)




    def genOpt_cycleParsGenOptimization(self,
                                        last_date = '2022-04-04',
                                        optimization_label = 'daily_middle_long_period',
                                        folder_path = None,
                                        file_name = None,
                                        num_samples_min = 600,
                                        num_samples_max = 1200,
                                        best_fit_start_back_period = None,
                                        final_kept_n_dominant_circles_min = 1,
                                        final_kept_n_dominant_circles_max = 15,
                                        min_period_min = 90,
                                        min_period_max = 120,
                                        max_period_min = 170,
                                        max_period_max = 270,
                                        hp_filter_lambda_min = 1,
                                        hp_filter_lambda_max = 2e9,
                                        hp_filter_lambda_n = 5000,
                                        periods_number = 180,
                                        opt_algo_type = 'genetic_omny_frequencies',  # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency'
                                        detrend_type = 'hp_filter', #'hp_filter', 'linear'
                                        windowing = None, # None 'kaiser',
                                        kaiser_beta = 3,
                                        period_related_rebuild_range = False,
                                        population_n = 100, # Population number
                                        NGEN = 10,  # Generations number
                                        CXPB = 0.7,  # Crossover probability
                                        MUTPB = 0.35,  # Mutation probability
                                        fitness_function = 'trading_pl', # 'trading_pl', 'mse'
                                        enabled_multiprocessing = True,
                                       ):
        
        """
This function optimize some hyperparameters necessary for the function multiperiod_analysis; the restults are saved in a CSV file in the folder data_storage_path defined at the moment of the class istantiation. 


Multirange Analysis Optimized parameters
------------------------------------------

- In case of detrend_type == hp_filter
    - num_samples
    - final_kept_n_dominant_circles
    - min_period, max_period (if not equal at origin)
    - hp_filter_lambda
    - period_related_rebuild_multiplier, if period_related_rebuild_range == True
    
- In case of detrend_type == linear
    - num_samples
    - final_kept_n_dominant_circles
    - min_period, max_period (if not equal at origin)
    - linear_filter_window_size_multiplier
    - period_related_rebuild_multiplier, if period_related_rebuild_range == True
    

Function parameters
---------------------

- optimization_label: textual label for identifying the sequence of optimization

- periods_number: number of past candles (periods) relative to the current date considered in the optimization to calculate the best_fit. The optimization procedure repeats the operation of extracting the dominant cycles and the best fit for each one for each period preceding the current one up to periods_number back. Then the average of the best fit over these periods_number periods is returned.

- min_period, max_period: minimum and maximum range of periods (inverse of frequency) considered in the transformation.

- final_kept_n_dominant_circles_min, final_kept_n_dominant_circles_max: range of number of dominant cycles considered for each range of frequency transformation.

- min_period_min, min_period_max: ranges of periods (frequencies) of min value transformation. Keep equal to have left period (1/frequency) limit fixed and not optimized

- max_period_min, max_period_max: ranges of periods (frequencies) of max value transformation. Keep equal to have right period (1/frequency) limit fixed and not optimized

- detrend_type: hp_filter, linear

- hp_filter_lambda: used only if detrend_type == "hp_filter", detrend index used in the hp filter; parameter to be optimized

- linear_filter_window_size_multiplier: used only if detrend_type == "linear", detrend index used in the linear filter; parameter to be optimized

- period_related_rebuild_range:

    - if True: in the signal reconstruction with dominant cycles from multiple period ranges, only the most recent portion of the signal is summed, where the calculation of how recent the portion of the current cycle to be summed should be is proportional to its period (frequency) and particularly, the proportionality is linked to the parameter period_related_rebuild_multiplier to be optimized. Also, in this case, num_samples must be optimized because it is also used as a range of values to be transformed.
    - if False: all dominant components are summed considering a number of periods equal to num_samples prior to the current date; if multiple frequency ranges are considered, each of them can be associated with an independent num_samples, which therefore becomes a parameter to be optimized. num_samples is also the number of samples used in the frequency transform and originally one must start from an index equal to the current date minus num_samples in reconstructing the signal for coherence with the phase returned by the transform. In this case, since period_related_rebuild_multiplier is not used, it does not need to be optimized.
period_related_rebuild_multiplier: only if period_related_rebuild_range == "True", in the signal reconstruction with dominant cycles, only the most recent portion of the signal is summed starting from current_component_period * max_period

- num_samples: number of samples considered in the frequency transform; in the multi-range analysis, it is one num_samples per range. It is a parameter to be optimized. Also, if period_related_rebuild_range = True, it also coincides with the number of samples of each component to be summed in the composite signal.

- best_fit_start_back_period: the fitness function evaluates a period equal to the longest rebuilt cycle if best_fit_start_back_period == None or 0; otherwise, it starts the comparison between the original detrended signal and the rebuilt one from the index len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period.

- opt_algo_type:

    - 'genetic_omny_frequencies': a genetic algorithm is used that tries to optimize by considering the dominant components from all frequency (period) ranges
    - 'genetic_frequencies_ranges': a genetic algorithm is used that tries to optimize by considering the dominant components from one frequency (period) range at a time
'mono_frequency': one frequency is optimized at a time starting from the lowest ones (longer periods); the optimization proceeds by keeping the amplitudes of the already optimized components fixed and adding the new component to the already fixed composite signal with the amplitude to be tested until finding the one that best fits the new composite signal.
    - 'tpe': the TPE algorithm is used on the components from all transformation ranges
    - 'atpe': the ATPE algorithm is used on the components from all transformation ranges


        """
        if(folder_path is None):
            folder_path = self.data_storage_path 
        
        if(file_name is None):
            file_name = self.ticker +" - tf " + self.state['data_timeframe'] + " - cyclces_analysis_hypeparmeters_optimization"

        
        run_start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        column_names = [
                    'analysis_reference_date',
                    'run_start_datetime',
                    'run_end_datetime',
                    'ticker_symbol',
                    'optimization_label',
                    'generation_number',
                    'fitness_function',
                    'detrend_type',
                    'windowing', 
                    'kaiser_beta',
                    'opt_algo_type',
                    'opt_pars_last_date',
                    'opt_pars_optimization_label',
                    'opt_pars_num_samples_min',
                    'opt_pars_num_samples_max',
                    'opt_pars_final_kept_n_dominant_circles_min',
                    'opt_pars_final_kept_n_dominant_circles_max',
                    'opt_pars_min_period_min',
                    'opt_pars_min_period_max',
                    'opt_pars_max_period_min',
                    'opt_pars_max_period_max',
                    'opt_pars_hp_filter_lambda_min',
                    'opt_pars_hp_filter_lambda_max',
                    'opt_pars_hp_filter_lambda_n',
                    'opt_pars_periods_number',
                    'opt_period_related_rebuild_range',
                    'opt_best_fit_start_back_period',
                    'opt_pars_population_n',
                    'opt_pars_NGEN',
                    'opt_pars_CXPB',
                    'opt_pars_MUTPB',
                    'best_individual_num_samples',
                    'best_individual_final_kept_n_dominant_circles',
                    'best_individual_min_period',
                    'best_individual_max_period',
                    'best_individual_linear_filter_window_size_multiplier',
                    'best_individual_period_related_rebuild_multiplier',
                    'best_individual_hp_filter_lambda',
                    'best_fitness_value_sum',
                    'best_fitness_max_loss',
                    'best_fitness_max_cumulative_loss',
                    'best_fitness_profits_sum',
                    'best_fitness_profits_count',
                    'best_fitness_losses_sum',
                    'best_fitness_losses_count',
                    'best_fitness_count_profits_vs_losses',
                    'best_fitness_profits_vs_losses'
                ]

        
        results_history = pd.DataFrame(columns=column_names)


        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.genOpt_last_date = last_date
        self.genOpt_num_samples_min = num_samples_min
        self.genOpt_num_samples_max = num_samples_max
        self.genOpt_final_kept_n_dominant_circles_min = final_kept_n_dominant_circles_min
        self.genOpt_final_kept_n_dominant_circles_max = final_kept_n_dominant_circles_max
        self.genOpt_min_period_min = min_period_min
        self.genOpt_min_period_max = min_period_max
        self.genOpt_max_period_min = max_period_min
        self.genOpt_max_period_max = max_period_max
        self.genOpt_periods_number = periods_number
        self.best_fit_start_back_period = best_fit_start_back_period
        self.opt_algo_type = opt_algo_type
        self.fitness_function = fitness_function
        self.detrend_type = detrend_type
        self.period_related_rebuild_range = period_related_rebuild_range
        self.windowing = windowing
        self.kaiser_beta = kaiser_beta



        # Build optimization domains according to detrending mode.
        
        if(self.period_related_rebuild_range == True): 
            
            self.log_info("Creating period-related rebuild range domain", function="genOpt_cycleParsGenOptimization")
        
            start = Decimal('2.0')
            stop = Decimal('6.05')
            step = Decimal('0.05')

            self.period_related_rebuild_multiplier_sequence = []
            current = start
            while current <= stop:
                self.period_related_rebuild_multiplier_sequence.append(float(current))
                current += step
    
            self.log_debug(
                "Period-related rebuild multiplier sequence created",
                function="genOpt_cycleParsGenOptimization",
                sequence=self.period_related_rebuild_multiplier_sequence,
            )
            
        if(self.detrend_type == 'hp_filter'):
            logarithmic_sequence = np.logspace(np.log10(hp_filter_lambda_min), np.log10(hp_filter_lambda_max), num=hp_filter_lambda_n, dtype=int)
            logarithmic_sequence = np.unique(logarithmic_sequence)
            self.genOpt_logarithmic_sequence = logarithmic_sequence
            
            self.log_debug(
                "HP lambda logarithmic sequence created",
                function="genOpt_cycleParsGenOptimization",
                hp_filter_lambda_min=hp_filter_lambda_min,
                logarithmic_sequence=logarithmic_sequence,
            )
            
            # Include period-related rebuild multiplier in the search domain.
            if(self.period_related_rebuild_range == True): 
                
                low=[num_samples_min,
                     final_kept_n_dominant_circles_min,
                     min_period_min,
                     max_period_min,
                     hp_filter_lambda_min,
                     2.0
                    ]

                up=[num_samples_max,
                    final_kept_n_dominant_circles_max,
                    min_period_max,
                    max_period_max,
                    hp_filter_lambda_max,
                    6.0
                   ]
                
            # Search only core cycle parameters when rebuild range is fixed.
            else:
            
                low=[num_samples_min,
                    final_kept_n_dominant_circles_min,
                    min_period_min,
                    max_period_min,
                    hp_filter_lambda_min]

                up=[num_samples_max,
                    final_kept_n_dominant_circles_max,
                    min_period_max,
                    max_period_max,
                    hp_filter_lambda_max]
        
        elif(self.detrend_type == 'linear'):
            
            self.linear_filter_window_size_multiplier_sequence = np.arange(0.5, 2.05, 0.05).tolist()
                        
            # Include period-related rebuild multiplier in the linear-detrend domain.
            if(self.period_related_rebuild_range == True): 
                
                low=[num_samples_min,
                    final_kept_n_dominant_circles_min,
                    min_period_min,
                    max_period_min,                    
                    0,
                    2.0]

                up=[num_samples_max,
                    final_kept_n_dominant_circles_max,
                    min_period_max,
                    max_period_max,                    
                    2.0,
                    6.0]
                
            # Search only core cycle parameters when rebuild range is fixed.
            else:
                
                low=[num_samples_min,
                     final_kept_n_dominant_circles_min,
                     min_period_min,
                     max_period_min,                    
                     0
                    ]

                up=[num_samples_max,
                    final_kept_n_dominant_circles_max,
                    min_period_max,
                    max_period_max,                    
                    2.0
                   ]               



        # Select the DEAP fitness shape according to the objective type.
        if(fitness_function == 'trading_pl'): # 'trading_pl'
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 0.4, 0.4, 1, 1, 0.6, -0.6, 1, 1)) # loss are always negative so they must be maximized
        else: # 'mse'
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, ))

        # Register DEAP individuals and genetic operators.
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Configure population generation, evaluation, crossover and mutation.
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.genOpt_initializeIndividual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        if(fitness_function == 'trading_pl'):
            toolbox.register("evaluate", self.genOpt_evaluateFitness)
        else: # 'mse'
            toolbox.register("evaluate", self.genOpt_evaluateMSEFitness)

        toolbox.register("mate", tools.cxTwoPoint)
        
        toolbox.register("mutate", tools.mutUniformInt,
                                   low=low,
                                   up=up, 
                                   indpb=0.2)
        
        toolbox.register("select", tools.selTournament, tournsize=3)

        if(enabled_multiprocessing == True):
            self.log_debug(
                "Configuring multiprocessing for genOpt",
                function="genOpt_cycleParsGenOptimization",
                start_method=multiprocessing.get_start_method(),
            )

            # Use spawn on Windows before registering pool.map.
            if multiprocessing.get_start_method() != 'spawn':
                self.log_debug("Setting multiprocessing start method to spawn", function="genOpt_cycleParsGenOptimization")
                multiprocessing.set_start_method('spawn')

            cpu_count = multiprocessing.cpu_count()
            self.log_debug("Multiprocessing pool configured", function="genOpt_cycleParsGenOptimization", cpu_count=cpu_count)

            # Enable multiprocessing for fitness evaluation.
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
        
        else:
            
            self.log_info("Multiprocessing disabled for genOpt", function="genOpt_cycleParsGenOptimization")


        # Create the initial population.
        population = toolbox.population(n=population_n)

        
        
        best_individual_num_samples = None
        best_individual_final_kept_n_dominant_circles = None
        best_individual_min_period = None
        best_individual_max_period = None
        best_individual_hp_filter_lambda = None
        best_individual_linear_filter_window_size_multiplier = None
        best_individual_period_related_rebuild_multiplier = None
        

        for gen in range(NGEN):

            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
            fits = toolbox.map(toolbox.evaluate, offspring)
            count = 0
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                count += 1
                self.log_debug(
                    "genOpt individual evaluated",
                    function="genOpt_cycleParsGenOptimization",
                    generation=gen + 1,
                    generations=NGEN,
                    population_index=count,
                    population_n=population_n,
                )
                
            population = toolbox.select(offspring, k=len(population))

            best_individual = tools.selBest(population, k=1)[0]
            best_fitness = best_individual.fitness.values
            
            self.log_info(
                "genOpt generation completed",
                function="genOpt_cycleParsGenOptimization",
                generation=gen + 1,
                best_fitness=best_fitness,
                best_individual=list(best_individual),
            )
            
            best_fitness_value_sum = best_fitness[0]
                
            if(fitness_function == 'trading_pl'):
                best_fitness_max_loss = best_fitness[1]
                best_fitness_max_cumulative_loss = best_fitness[2]
                best_fitness_profits_sum =  best_fitness[3]
                best_fitness_profits_count =best_fitness[4]
                best_fitness_losses_sum = best_fitness[5]
                best_fitness_losses_count = best_fitness[6]
                best_fitness_count_profits_vs_losses = best_fitness[7]
                best_fitness_profits_vs_losses = best_fitness[8]
            else:
                best_fitness_max_loss = None
                best_fitness_max_cumulative_loss = None
                best_fitness_profits_sum =  None
                best_fitness_profits_count = None
                best_fitness_losses_sum = None
                best_fitness_losses_count = None
                best_fitness_count_profits_vs_losses = None
                best_fitness_profits_vs_losses = None
                
            if(detrend_type == 'hp_filter'):
                best_individual_num_samples = best_individual[0]
                best_individual_final_kept_n_dominant_circles = best_individual[1]
                best_individual_min_period = best_individual[2]
                best_individual_max_period = best_individual[3]
                best_individual_hp_filter_lambda = best_individual[4]
                
                
            elif(detrend_type == 'linear'):
                best_individual_num_samples = best_individual[0]
                best_individual_final_kept_n_dominant_circles = best_individual[1]
                best_individual_min_period = best_individual[2]
                best_individual_max_period = best_individual[3]
                best_individual_linear_filter_window_size_multiplier = best_individual[4]
                
                
            if(self.period_related_rebuild_range == True):
                best_individual_period_related_rebuild_multiplier = best_individual[5]
                
            else:
                best_individual_period_related_rebuild_multiplier = 'Default'
                

            results_history = pd.DataFrame(columns=column_names) # reset dataframe, just one line per time updated in the csv file
            new_row = pd.Series({
                'analysis_reference_date': last_date,
                'run_start_datetime': run_start_datetime,
                'run_end_datetime': np.nan,
                'ticker_symbol': self.ticker,
                'optimization_label': optimization_label,
                'generation_number': gen + 1,                
                'fitness_function': self.fitness_function,
                'detrend_type': self.detrend_type,
                'windowing': self.windowing, 
                'kaiser_beta': self.kaiser_beta,
                'opt_algo_type': self.opt_algo_type,
                'opt_pars_last_date': last_date,
                'opt_pars_optimization_label': optimization_label,
                'opt_pars_num_samples_min': num_samples_min,
                'opt_pars_num_samples_max': num_samples_max,
                'opt_pars_final_kept_n_dominant_circles_min': final_kept_n_dominant_circles_min,
                'opt_pars_final_kept_n_dominant_circles_max': final_kept_n_dominant_circles_max,
                'opt_pars_min_period_min': min_period_min,
                'opt_pars_min_period_max': min_period_max,
                'opt_pars_max_period_min': max_period_min,
                'opt_pars_max_period_max': max_period_max,
                'opt_pars_hp_filter_lambda_min': hp_filter_lambda_min,
                'opt_pars_hp_filter_lambda_max': hp_filter_lambda_max,
                'opt_pars_hp_filter_lambda_n': hp_filter_lambda_n,
                'opt_pars_periods_number': periods_number,
                'opt_period_related_rebuild_range': self.period_related_rebuild_range, 
                'opt_best_fit_start_back_period': self.best_fit_start_back_period, 
                'opt_pars_population_n': population_n,
                'opt_pars_NGEN': NGEN,
                'opt_pars_CXPB': CXPB,
                'opt_pars_MUTPB': MUTPB,
                'best_individual_num_samples': best_individual_num_samples,
                'best_individual_final_kept_n_dominant_circles': best_individual_final_kept_n_dominant_circles,
                'best_individual_min_period': best_individual_min_period,
                'best_individual_max_period': best_individual_max_period,
                'best_individual_linear_filter_window_size_multiplier': best_individual_linear_filter_window_size_multiplier, 
                'best_individual_period_related_rebuild_multiplier': best_individual_period_related_rebuild_multiplier,                
                'best_individual_hp_filter_lambda': best_individual_hp_filter_lambda,
                'best_fitness_value_sum': best_fitness_value_sum,
                'best_fitness_max_loss': best_fitness_max_loss,
                'best_fitness_max_cumulative_loss': best_fitness_max_cumulative_loss,
                'best_fitness_profits_sum': best_fitness_profits_sum,
                'best_fitness_profits_count': best_fitness_profits_count,
                'best_fitness_losses_sum': best_fitness_losses_sum,
                'best_fitness_losses_count': best_fitness_losses_count,
                'best_fitness_count_profits_vs_losses': best_fitness_count_profits_vs_losses,
                'best_fitness_profits_vs_losses': best_fitness_profits_vs_losses
            })


            results_history.loc[len(results_history)] = new_row
            
            self.save_dataframe(dataframe = results_history, folder_path = folder_path, file_name = file_name)

            if(self.output_clearing == True):
                clear_output(wait=True)
                
            if self.is_log_enabled("DEBUG"):
                display(results_history)


        if(enabled_multiprocessing == True):
            pool.close()
            pool.join()


        run_end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        final_history = self.save_dataframe(dataframe = results_history,
                            folder_path = folder_path,
                            file_name = file_name,
                            update_column = True,
                            update_column_name = 'run_end_datetime',
                            update_column_value = run_end_datetime,
                            filter_column_name = 'run_start_datetime',
                            filter_column_value = run_start_datetime)

        self.log_info("genOpt final results saved", function="genOpt_cycleParsGenOptimization")
        
        if(self.output_clearing == True):
            clear_output(wait=True)
            
        if self.is_log_enabled("DEBUG"):
            display(final_history)



        return final_history
