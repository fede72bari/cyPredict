import yfinance as yf

# Basics
from enum import Enum
from typing import Type
import sys
import traceback
# import goertzel # C dll
from goertzel import goertzel_general_shortened as goertzel_general_shortened
from goertzel import goertzel_DFT as goertzel_DFT
import genetic_optimization # C dll
from genetic_optimization import evaluate_fitness as evaluate_fitness
import time
from decimal import Decimal


# Math, sci and stats libraries
# import numpy as np
# import math
# from scipy.signal import argrelextrema
# import random
# from scipy.signal import find_peaks
# from scipy.signal import argrelmax, argrelmin
# from sklearn.preprocessing import StandardScaler
# from scipy.integrate import simps
# from scipy.stats import pearsonr, spearmanr, kendalltau
# from scipy.integrate import simps
# from scipy.signal import savgol_filter
# from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import cdist
# from scipy.spatial.distance import euclidean
from statsmodels.nonparametric.smoothers_lowess import lowess


import numpy as np
import math
from scipy.signal import argrelextrema
import random
import scipy
from scipy.signal import find_peaks
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.integrate import simps
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from statsmodels.tsa.tsatools import detrend
import random


# Optimization libraries
from deap import base, creator, tools, algorithms
from hyperopt import fmin, tpe, hp, atpe
from hyperopt.pyll import scope


# Data Management
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Charting and reporting libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tabulate import tabulate
from IPython.display import display
from IPython.display import clear_output
from tabulate import tabulate
from pprint import pprint
import plotly.io as pio
pio.renderers.default='notebook'

# Financial stats and indicators
import talib

# Stings manipulation
import re

# Filse manipulation
import os

# Processing capability
import multiprocessing

# Time Management
import datetime
from datetime import datetime, timedelta, date
import pytz
from pytz import timezone



class Drive(Enum):
    local = 1
    GoogleDrive = 2

class financialDataSource(Enum):
    csv_file = 1
    yfinance = 2

class cyPredict:

    def __init__(self,
                 data_source="yfinance",
                 ticker="SPY",
                 data_start_date="2004-01-01",
                 data_end_date=None,
                 data_timeframe="1d",
                 data_storage_path="\\cyPredict\\",
                 time_tracking = False,
                 output_clearing = False,
                 print_activity_remarks = True): 

        # Instance variables (attributes)
        self.data_source = data_source
        self.ticker = ticker
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.data = []
        self.data_storage_path = data_storage_path
        self.print_activity_remarks = print_activity_remarks
        

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
#         self.MultiAn_dominant_cycles_df = pd.DataFrame(columns=['peak_frequencies', 'peak_phases', 'start_rebuilt_signal_index'])
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
            "ticker": ticker,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "data_timeframe": data_timeframe,
            "data_state": 'not initialized',  # error, initialized
            "data_state_msg": '',
        }

        self.download_finance_data(
            self.state["data_source"],
            self.state["ticker"],
            self.state["data_start_date"],
            self.state["data_end_date"],
            self.state["data_timeframe"]
        )

        self.scaler = MinMaxScaler(feature_range=(-1, 1)) #StandardScaler()
        
        self.time_tracking = time_tracking
        self.start_time = time.time()
        self.end_time = time.time()
        
        self.output_clearing = output_clearing
        
    def track_time(self, message):
        
        if(self.time_tracking == True):
            self.end_time = time.time()        
            print(f'{message}, delta time: {self.end_time - self.start_time}')        
            self.start_time = time.time()
            
    def set_start_time(self):

        self.start_time = time.time()
        

    def download_finance_data(
        self,
        data_source,
        ticker,
        data_start_date,
        data_end_date,
        data_timeframe
    ):
        try:
            if data_source == 'yfinance':
                self.data = yf.download(ticker, start=data_start_date, end=data_end_date, interval=data_timeframe)
                self.data = self.data.xs(ticker, level=1, axis=1)

                if(not self.data.empty):
                    self.state["data_state"] = 'initialized'
                    self.state["data_state_msg"] = 'Financial data is ready to be used.'
                else:
                    self.state["data_state"] = 'error'
                    self.state["data_state_msg"] = data_source + ' returned empty data, no error speficiation returned by the module. Look at standard output.'

            else:
                self.state["data_state"] = 'error'
                self.state["data_state_msg"] = 'Not managed data source ' + data_source
                print('Error: not managed data source ' + data_source)
        except Exception as e:  # ExceptionType should be Exception
            self.state["data_state"] = 'error'
            self.state["data_state_msg"] = str(e)
            print(f"An error occurred in download_finance_data: {e}")


#     def goertzel_general_shortened(self, x, indvec):
#         # Check input arguments
#         if len(indvec) < 1:
#             raise ValueError('Not enough input arguments')
#         if not isinstance(x, np.ndarray) or x.size == 0:
#             raise ValueError('X must be a nonempty numpy array')
#         if not isinstance(indvec, np.ndarray) or indvec.size == 0:
#             raise ValueError('INDVEC must be a nonempty numpy array')
#         if np.iscomplex(indvec).any():
#             raise ValueError('INDVEC must contain real numbers')

#         lx = len(x)
#         x = x.reshape(lx, 1)  # forcing x to be a column vector

#         # Initialization
#         no_freq = len(indvec)
#         y = np.zeros((no_freq,), dtype=complex)

#         # Computation via second-order system
#         for cnt_freq in range(no_freq):
#             # Precompute constants
#             pik_term = 2 * np.pi * indvec[cnt_freq] / lx
#             cos_pik_term2 = 2 * np.cos(pik_term)
#             cc = np.exp(-1j * pik_term)  # complex constant

#             # State variables
#             s0 = 0
#             s1 = 0
#             s2 = 0

#             # Main loop
#             for ind in range(lx - 1):
#                 s0 = x[ind] + cos_pik_term2 * s1 - s2
#                 s2 = s1
#                 s1 = s0

#             # Final computations
#             s0 = x[lx - 1] + cos_pik_term2 * s1 - s2
#             y[cnt_freq] = s0 - s1 * cc

#             # Complex multiplication substituting the last iteration
#             # and correcting the phase for potentially non-integer valued
#             # frequencies at the same time
#             y[cnt_freq] = y[cnt_freq] * np.exp(-1j * pik_term * (lx - 1))

#         return y



    def hp_filter(self, data, lambda_, ret=False):
        nobs = len(data)
        output = np.copy(data)

        if nobs <= 5:
            print('nobs <= 5')
            return None, 0  # Not enough data

        a = np.zeros(nobs)
        b = np.zeros(nobs)
        c = np.zeros(nobs)

        a[0] = 1.0 + lambda_
        b[0] = -2.0 * lambda_
        c[0] = lambda_

        for i in range(1, nobs - 2):
            a[i] = 6.0 * lambda_ + 1
            b[i] = -4.0 * lambda_
            c[i] = lambda_

        a[1] = 5.0 * lambda_ + 1
        a[nobs - 1] = 1.0 + lambda_
        a[nobs - 2] = 5.0 * lambda_ + 1.0

        b[nobs - 2] = -2.0 * lambda_

        H1 = 0
        H2 = 0
        H3 = 0
        H4 = 0
        H5 = 0
        HH1 = 0
        HH2 = 0
        HH3 = 0
        HH4 = 0
        HH5 = 0
        Z = 0
        HB = 0
        HC = 0

        for i in range(nobs):
            Z = a[i] - H4 * H1 - HH5 * HH2
            if Z == 0:
                return None, 3  # Division by zero
            HB = b[i]
            HH1 = H1
            H1 = (HB - H4 * H2) / Z
            b[i] = H1
            HC = c[i]
            HH2 = H2
            H2 = HC / Z
            c[i] = H2
            a[i] = (output[i] - HH3 * HH5 - H3 * H4) / Z
            HH3 = H3
            H3 = a[i]
            H4 = HB - H5 * HH1
            HH5 = H5
            H5 = HC

        H2 = 0
        H1 = a[nobs - 1]
        output[nobs - 1] = H1

        for i in range(nobs - 1, 0, -1):
            output[i - 1] = a[i - 1] - b[i - 1] * H1 - c[i - 1] * H2
            H2 = H1
            H1 = output[i - 1]

        if not ret:
            output = data - output

        return output, 1



    def get_bartels_score(self, dataset, cycle_length, max_segments):
        bartelsscore = 0
        segmentspassed = 0

        datacounter = 0
        A = 0
        B = 0
        SUM_A = 0
        SUM_B = 0
        SUM_A2B2 = 0
        bval = 0
        SI = 0
        CO = 0
        bogenmass = 0

        bval = 360 / cycle_length

        for x in range(len(dataset)):
            bogenmass = (bval * (x + 1)) / 180 * math.pi

            SI = math.sin(bogenmass) * dataset[x]
            CO = math.cos(bogenmass) * dataset[x]

            A += SI
            B += CO

            datacounter += 1

            if datacounter == int(cycle_length) and segmentspassed < int(max_segments):
                SUM_A += A
                SUM_B += B
                SUM_A2B2 += A ** 2 + B ** 2

                segmentspassed += 1
                datacounter = 0
                A = 0
                B = 0

        if segmentspassed == 0:
            return_bartels = 0
            return_segments = 0
        else:
            SUM_A_Average = SUM_A / segmentspassed
            SUM_B_Average = SUM_B / segmentspassed
            Amplitude = math.sqrt(SUM_A_Average ** 2 + SUM_B_Average ** 2)

            SUM_A2B2_Average = SUM_A2B2 / segmentspassed
            Amplitude_A2B2 = math.sqrt(SUM_A2B2_Average)

            a1 = Amplitude_A2B2 / math.sqrt(segmentspassed)
            b1 = Amplitude / a1

            bartelsscore = 1 / math.exp(b1 ** 2)

        return bartelsscore, segmentspassed



    def jh_filter(self, y, p = 4, h = 8):
        n = len(y)

        print("p: " + str(p) + ", h: " +str(h))
        # h = 8  # Choose the desired forecast horizon

        # Initialize arrays for regression inputs and outputs
        X = np.ones((n - h, p + 1))
        y_est = np.zeros(n - h)

        # Fill in the arrays
        for i in range(p):
            X[:, i+1] = y[i:n-h+i]
        for i in range(n - h):
            y_est[i] = y[i+h]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y_est)

        # Calculate the estimated cyclical component
        cyclical_component = model.predict(X)

        # Calculate residuals (detrended data)
        detrended_y = y[h:] - cyclical_component

        return detrended_y



    def datetime_dateset_extend(self, df, extension_periods = 10):

        df.index = df.index.tz_localize(None)

        today = pd.to_datetime(df.index.max().date())


        # Yesterday data
        #   in yahoo df yesterday data could be incompleted, refers to the available
        #   previous day before yesterday considering weekends absence of data
        yesterday = today - pd.DateOffset(days=1)
        yesterday_date = yesterday.date()  # Converti Timestamp in datetime.date
        before_yesterday_data = df[df.index.date < yesterday_date]
        last_complete_day_date = pd.Timestamp(before_yesterday_data.index.max()).date() #pd.to_datetime(before_yesterday_data.index.max().date())
        # last_complete_day_data = before_yesterday_data[before_yesterday_data.index.date == last_complete_day_date]
        last_complete_day_data = before_yesterday_data[before_yesterday_data.index.date == last_complete_day_date] #pd.Timestamp(last_complete_day_date).date()]
        
#         print(f'yesterday_date = {yesterday_date}')
#         print(f'last_complete_day_date = {last_complete_day_date}')
#         print(f'last_complete_day_data = {last_complete_day_data}')


        # List of the yesterday sample times
#         yesterday_times = last_complete_day_data.index.to_series().apply(lambda x: (x.hour, x.minute, x.second))

        yesterday_times = pd.DataFrame({
            'hour': last_complete_day_data.index.hour,
            'minute': last_complete_day_data.index.minute,
            'second': last_complete_day_data.index.second
        })

        samples_per_day = len(yesterday_times)
        
#         print('\tQUI C1')

        # Inizialization
        new_datetime = today
        new_indexes = []

        last_datetime = df.index[-1]
        last_time = (last_datetime.hour, last_datetime.minute, last_datetime.second)
#         time_cardinality = yesterday_times.tolist().index(last_time)
        # Confronta le colonne di yesterday_times con last_time per trovare l'indice corrispondente
#         print(f'yesterday_times = {yesterday_times}')
#         print(f'last_time = {last_time}')
        time_cardinality_row = (yesterday_times['hour'] == last_time[0]) & (yesterday_times['minute'] == last_time[1]) &                            (yesterday_times['second'] == last_time[2])

        # Trova l'indice effettivo
        time_cardinality = time_cardinality_row.idxmax()


        for _ in range(extension_periods):
            

            # remain in the same day
            if(time_cardinality < (samples_per_day-1)):
                time_cardinality += 1

            else:
                # restore to first sample time of the day
                time_cardinality = 0

                # increase by one day skeeping weekends
                days = 1

                while True:
                    temp_date = new_datetime + pd.DateOffset(days=days)
                    if temp_date.weekday() < 5:  # Working day
                        break
                    days += 1

                new_datetime = temp_date

#             new_datetime = new_datetime.replace(hour = yesterday_times[time_cardinality].iloc[0],
#                                             minute = yesterday_times[time_cardinality].iloc[1],
#                                             second = yesterday_times[time_cardinality].loc[2])

            
            # Supponendo che time_cardinality sia l'indice della riga che vuoi utilizzare
            new_hour = yesterday_times.loc[time_cardinality, 'hour']
            new_minute = yesterday_times.loc[time_cardinality, 'minute']
            new_second = yesterday_times.loc[time_cardinality, 'second']

            # Aggiornamento dell'oggetto datetime new_datetime con i nuovi valori di hour, minute, second
            new_datetime = new_datetime.replace(hour=new_hour, minute=new_minute, second=new_second)


            new_indexes.append(new_datetime)

        # Extend the original dataframe with new rows and datetime indexes
        new_rows = pd.DataFrame(np.nan, index=new_indexes, columns=df.columns)
        df = pd.concat([df, new_rows])

        return df

#     def linear_detrend(self, data, window_size = 0):
        
#         if(window_size == 0):
#             break_points = 0
            
#         else:
            
#             num_complete_windows = len(data) // window_size
#             initial_window_size = len(data) % window_size if len(data) % window_size != 0 else window_size
#             break_points = [initial_window_size] + [initial_window_size + i * window_size for i in range(1, num_complete_windows + 1)]

#         detrend = scipy.signal.detrend(data, type='linear', bp=break_points)
        
#         return pd.Series(detrend, index=data.index)

    def linear_detrend(self, data, window_size=0):

        if window_size == 0:
            break_points = 0
        else:
            total_length = len(data)
            remainder = total_length % window_size
            start_index = remainder  # Inizia dal punto in cui inizia la prima finestra completa
            
            num_complete_windows = (total_length - remainder) // window_size
            break_points = [start_index + i * window_size for i in range(1, num_complete_windows)]
            
            # Assicurarsi che tutti i breakpoints siano validi
            break_points = [bp for bp in break_points if bp < total_length]

        
        # Applicare il detrend solo alla porzione dei dati con finestre complete
        detrended_data = scipy.signal.detrend(data[start_index:], type='linear', bp=[bp - start_index for bp in break_points])
        
        # Combina i dati esclusi inizialmente con i dati detrendati per mantenere la stessa lunghezza
        detrended_data_full = np.concatenate((data[:start_index], detrended_data))
        
        return pd.Series(detrended_data_full, index=data.index)




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
                         lowess_k = 3,
                         centered_averages = True,
                         time_zone = None,
                         other_correlations = False,
                         include_calibrated_MACD = False,
                         include_calibrated_RSI = False,
                         show_charts = False,
                         print_report = True,
                         indicators_signal_calcualtion = True,
                         debug = False,
                         enabled_multiprocessing = False,
                         time_tracking = False                         
                         ):

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
          "lowess_k": lowess_k
        }

        
        
        if(time_tracking):
            
            self.set_start_time()
            self.track_time('\nTime tracking started.')

        

        signals_results = pd.DataFrame()
        # signals_results['parameters'] = configuration

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

        if(time_zone != None):
            time_zone = timezone(time_zone)


        # Original_data index type conversion to DateTime
        original_data.index = pd.to_datetime(original_data.index)
        original_data = original_data.sort_index()


        # ------------------------------------------------------
        #      provided start date and num of samples
        # ------------------------------------------------------

        # Find last element of the freqnecy transofrmation range
        if((start_date != None) & (num_samples != None)):

          # Types conversion

            if(time_zone != None):
                start_date = pd.Timestamp(start_date, tz=time_zone)

            else:
                start_date = pd.to_datetime(start_date)

            filtered_data_sd = original_data[original_data.index == start_date]

            if (filtered_data_sd.empty):

                filtered_data_sd = original_data[original_data.index.date == start_date.date()]
                print('Start datetime not found, attempt just with date.')

            # Find element with lowest time within start date
            if not filtered_data_sd.empty:

                min_datetime = filtered_data_sd.index.min()
                index_of_min_time_for_sd = original_data.index.get_loc(min_datetime)
                data = original_data[original_data.index.date >= pd.to_datetime(start_date).date()]
                data = data.head(num_samples)
                start_rebuilt_signal_index = index_of_min_time_for_sd
                end_rebuilt_signal_index = index_of_min_time_for_sd + num_samples
                print("Index of first smaple for start date (" + str(start_date) + "): " + str(index_of_min_time_for_sd))

            else:

                print(f"No data for start datetime equal to {start_date}.")
                return None, None, None, None, None


        # ------------------------------------------------------
        #      provided current date and num of samples
        # ------------------------------------------------------

        # Find first element of the freqnecy transofrmation range
        if((current_date != None)  & (num_samples != None)):


            # Types conversion
            if(time_zone != None):
                current_date = pd.Timestamp(current_date, tz=time_zone)

            else:
                current_date = pd.to_datetime(current_date)

            filtered_data_cd = original_data[original_data.index == current_date]


            if (filtered_data_cd.empty):

                filtered_data_cd = original_data[original_data.index.date == current_date.date()]

            # Find element with highest time within current date
            if not filtered_data_cd.empty:

                max_datetime = filtered_data_cd.index.max()
                # print('MAX DATETIME: ' + str(max_datetime))
                index_of_max_time_for_cd = original_data.index.get_loc(max_datetime)
                # print('index_of_max_time_for_cd: ' + str(index_of_max_time_for_cd))
                if(time_zone == None):
                    data = original_data[original_data.index <= current_date]

                if time_zone != None or data.empty:
                    data = original_data[original_data.index.date <= current_date.date()]

                data = data.tail(num_samples)


                start_rebuilt_signal_index = index_of_max_time_for_cd - num_samples + 1
                end_rebuilt_signal_index = index_of_max_time_for_cd

                if data.empty:
                    print(f"No data for current datetime equal to {current_date}. Possible not existing date (weekend?), wrong date format or mismatch with the timezone.")
                    
                    sys.exit("No data for current datetime")

                    return None, None, None, None, None

            else:
                print(f"No data for current datetime equal to {current_date}.")
                sys.exit("No data for current datetime")
                return None, None, None, None, None


        # ------------------------------------------------------
        #      provided start date and current date
        # ------------------------------------------------------


        # Number of elements in the provided range of dates
        if((num_samples == None) &
           (start_date != None) &
           (current_date != None)):

            if(time_zone != None):
                start_date = pd.Timestamp(start_date, tz=time_zone)

            else:
                start_date = pd.to_datetime(start_date)

            filtered_data_sd = original_data[original_data.index == start_date]


            if (filtered_data_sd.empty):

                filtered_data_sd = original_data[original_data.index.date == start_date.date()]
                print('Start datetime not found, attempt just with date.')

            if(time_zone != None):
                current_date = pd.Timestamp(current_date, tz=time_zone)

            else:
                current_date = pd.to_datetime(current_date)

            filtered_data_cd = original_data[original_data.index == current_date]

            if (filtered_data_cd.empty):

                current_date = current_date.date()

                filtered_data_cd = original_data[original_data.index.date == current_date]
                print('Current datetime not found, attempt just with date.')


            # Find element with highest time within current date
            if ((not filtered_data_sd.empty) & (not filtered_data_cd.empty)):

                min_datetime = filtered_data_sd.index.min()
                index_of_min_time_for_sd = original_data.index.get_loc(min_datetime)

                max_datetime = filtered_data_cd.index.max()
                index_of_max_time_for_cd = original_data.index.get_loc(max_datetime)

                data = original_data[(original_data.index.date >= pd.to_datetime(start_date).date()) &
                                   (original_data.index.date <= pd.to_datetime(current_date).date())]

                num_samples = index_of_max_time_for_cd - index_of_min_time_for_sd + 1

                start_rebuilt_signal_index = index_of_min_time_for_sd
                end_rebuilt_signal_index = index_of_max_time_for_cd

                print(f"Index of start date {start_date}: {start_rebuilt_signal_index}, index of current date {current_date}: {end_rebuilt_signal_index}.")

            else:
                print(f"No data between datetime {start_date} and date {current_date}.")
                return None, None, None, None, None


        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tData segment extracted, start data preparation')


        # ------------------------------------------------------
        #      Data preparation
        # ------------------------------------------------------


        # Transform frequencies range
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

        # Specific data column
        data = data[data_column_name].values

        # print('\n Data: ' + str(data[-20:]))

# MODIFICA DETRENDED 1: usare dati non tagliati, original_data invece di data

        # Replace Nan before detrending otherwise some detrending functions
        # will return a entire set of Nan

        original_data[data_column_name] = original_data[data_column_name].fillna(0)

        # Type of detrend technique
        if(detrend_type == 'linear'):
            print(f'linear detrend, detrend window = {detrend_window}')
            print(f'len orginal_data[data_column_name] = {len(original_data[data_column_name])}')
            # detrended_data = detrend(data, order=1)
#             detrended_data = detrend(original_data[data_column_name], order=1)

            detrended_data = self.linear_detrend(original_data[data_column_name], window_size = detrend_window)
            

        if(detrend_type == 'quadratic'):
            # print('quadratic detrend')
            # detrended_data = detrend(data, order=2)
            detrended_data = detrend(original_data[data_column_name], order=2)

        if(detrend_type == 'hp_filter'):
            # print('hp_filter')
            # detrended_data, _ = self.hp_filter(data, hp_filter_lambda)
            detrended_data, _ = self.hp_filter(original_data[data_column_name], hp_filter_lambda)


        if(detrend_type == 'jh_filter'):
            # print('jh_filter')
            # detrended_data = self.jh_filter(data, jp_filter_p, jp_filter_h)
            detrended_data = self.jh_filter(original_data[data_column_name], jp_filter_p, jp_filter_h)
            
            
        if(detrend_type == 'lowess'):
            _, detrended_data = self.detrend_lowess(original_data[data_column_name], max_period, k=4)
            
#         print(f'detrend data type {type(detrended_data)}')
#         print(detrended_data.iloc[0:10])

        # print('detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1]: ' + str(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1]))

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tData preparation eneded, start Goertzel')
        
        if(time_tracking):
            self.track_time('\tPartial time Data Preparation')


        # ------------------------------------------------------
        #      Goertzel DFT calculation
        # ------------------------------------------------------


        # Classic Goertzel DFT
        harmonics_amplitudes = []
        phases = []
        minoffset = []
        maxoffset = []
        frequency_range = frequency_range/num_samples


        for frequency in frequency_range:
# MODIFICA DETRENDED 2: prendere solo la parte di interesse quella coincidendte con gli indici con cui si estratta data
            # temp_amplitudes, temp_phases, _, temp_minoffset, temp_maxoffset = self.goertzel_DFT(detrended_data, 1/frequency)
#             if(debug == True):
# #                 print('\nfrequency: ' +  str(frequency))
# #                 print('\tlen detrended_data: ' +  str(len(detrended_data)))
# #                 print('\tstart_rebuilt_signal_index: '+ str(start_rebuilt_signal_index))
# #                 print('\tend_rebuilt_signal_index: ' + str(end_rebuilt_signal_index))
# #                 print("\tdetrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1].isnull().any(): ")
# #                 print(str(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1].isnull().any()))
# #                 print("\tDETRENDED DATA:")
# #                 display(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1])
# #                 print('\tNOT DETRENDED DATA:')
# #                 display(original_data[data_column_name].iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1])
# #                 print("\toriginal_data[data_column_name].isnull().any()")
#                 print(str(original_data[data_column_name].isnull().any()))

#             print("Tipo di 1/frequency:", type(1/frequency))
#             display(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1])
#             print(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1][data_column_name].dtype)


            temp_amplitudes, temp_phases, _, temp_minoffset, temp_maxoffset = goertzel_DFT(detrended_data.iloc[start_rebuilt_signal_index:end_rebuilt_signal_index+1], 1/frequency)

          # print('temp_amplitudes, temp_phases, temp_minoffset period' + str(temp_amplitudes) + ' ' + str(temp_phases) + ' ' +str(temp_minoffset) + ' ' + str(1/frequency))
            harmonics_amplitudes.append(temp_amplitudes)
            phases.append(temp_phases)
            minoffset.append(temp_minoffset)
            maxoffset.append(temp_maxoffset)

        minoffset = [int(x) for x in minoffset]
        maxoffset = [int(x) for x in maxoffset]

        harmonics_amplitudes = np.array(harmonics_amplitudes)
        phases = np.array(phases)
        # frequency_range = frequency_range/num_samples


        # Amplitude peaks extraction
        goertzel_df_peaks = pd.DataFrame()
        peaks_indexes = argrelmax(harmonics_amplitudes, order = 10)[0] # find indexes of peaks

        peak_frequencies = np.array(frequency_range[peaks_indexes])
        peak_periods = np.array(1 / frequency_range[peaks_indexes])
        peak_amplitudes = harmonics_amplitudes[peaks_indexes]
        peak_phases = phases[peaks_indexes]
        peak_next_min_offset = np.array(minoffset)[peaks_indexes]
        peak_next_max_offset = np.array(maxoffset)[peaks_indexes]


        # Amplitude scaling with respect to the frequency for dominant circles determination
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


        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tGoertzel finished, start harmonics ahanlysis')


        if(time_tracking):
            self.track_time('\tPartial time Goertzel Transform')
            
            
        # ------------------------------------------------------
        #      Limit the considered original harmonics
        # ------------------------------------------------------
        # Keep harmonics with period within min_period and max_period or the first
        # final_kept_n_dominant_circles ones
        cut_peaks_indexes = []

        # print(f'\t\t\t\tlen peaks_indexes: {peaks_indexes}')

        if(limit_n_harmonics != None):
#             print("limit_n_harmonics: " + str(limit_n_harmonics))
            cut_peaks_indexes = peaks_indexes[0:limit_n_harmonics]

        elif((min_period != None) & (max_period != None)):

#             print("\t\t\t\tlimit_n_harmonics: False")
#             print(f"\t\t\t\tmin_period: {min_period}, max_period: {max_period},")
#             print(f'\t\t\t\t\tpeak_periods: {peak_periods}')

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
        
#         print(f"\t\t\t\tlen(cut_peaks_indexes) == {len(cut_peaks_indexes)}")
        

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tHarmonixs ennded, started Bertel')

                            
        if(time_tracking):
            self.track_time('\tPartial time Partial time Limit N of harmonics')
        

        # ------------------------------------------------------
        #      Bartel peak scores
        # -----------------------------------------------------

        temp_filtered_f_indexes = np.array([], dtype=int)
        goertzel_df_peaks['bartel_score'] = np.nan

        if(bartel_peaks_filtering == True):
            for index in cut_peaks_indexes:
                frequency = frequency_range[index]
                cycle_length = 1 / frequency
                # max_segments = int(num_samples / cycle_length)
                divisor = 100 #16
                max_segments = 30 # int(num_samples/divisor)

                bartelsscore, _ = self.get_bartels_score(data, cycle_length, max_segments) #get_bartels_score(detrended_data, cycle_length, max_segments)
                # goertzel_df_peaks[goertzel_df_peaks['peaks_indexes'] == index]['bartel_score'] =  bartelsscore
                goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'bartel_score'] = bartelsscore

                if(bartelsscore >= bartel_scoring_threshold):
                    temp_filtered_f_indexes = np.append(temp_filtered_f_indexes, int(index))

            dominant_peaks_indexes = temp_filtered_f_indexes
        else:
            dominant_peaks_indexes = peaks_indexes


        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tBartel ended, started Correlations')

        if(time_tracking):
            self.track_time('\tPartial time Bartel Score')                           


        # ------------------------------------------------------
        #      Other Correlations
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


                # -------------------------------------------------------------
                #         Correlation between current scaled harmonic and
                #         scaled delta of long an short savgol_filtered signal
                # -------------------------------------------------------------

                if(debug == True):
                    print("Other correlations, period: " + str(period))
                    print("Other correlations, int(period*2): " + str(int(period*2)))
                    print("len(data_df[data_column_name]): " + str(len(data_df[data_column_name])))

                averages['savgol_filter_long'] = savgol_filter(data_df[data_column_name], int(period*2), 2)
                averages['savgol_filter_short'] = savgol_filter(data_df[data_column_name], int(period), 2)
                averages['savgol_filter_delta'] = averages['savgol_filter_short']  - averages['savgol_filter_long']
                averages['scaled_savgol_filter_delta'] = scaler.fit_transform(averages['savgol_filter_delta'].values.reshape(-1, 1)).flatten()
#                 print(f"Length of data_df[data_column_name]: {len(data_df[data_column_name])}")
#                 print(f"Length of signal['scaled_signal']: {len(signal['scaled_signal'])}")
#                 print(f"Length of averages['scaled_savgol_filter_delta']: {len(averages['scaled_savgol_filter_delta'])}")
#                 print(f"Period: {period}, Long filter period: {int(period*2)}, Short filter period: {int(period)}")
#                 print(f"Length of savgol_filter_long: {len(averages['savgol_filter_long'])}")
#                 print(f"Length of savgol_filter_short: {len(averages['savgol_filter_short'])}")
#                 print(f"Length of savgol_filter_delta: {len(averages['savgol_filter_delta'])}")



                goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_savgol_filter_delta_correlation'] = simps(signal['scaled_signal'] * np.roll(averages['scaled_savgol_filter_delta'], tau), dx=1)


            # if(data_column_name == 'RSI_column_name'):
            # print('\t\t\tCorrelations ended, started peaks analysis')
                            
                            
            if(time_tracking):
                self.track_time('\tPartial time Other Correlations')      

            # ------------------------------------------------------------------------------------
            #         Peaks cardinality error between
            #         current scaled harmonic and scaled_savgol_filter_delta_correlation
            # -------------------------------------------------------------------------------------


            # Peaks cardinality coherence and peak phase error between current scaled harmonic and scaled_average_delta
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

            # print("Period: " + str(period) + "peaks_tollerance: " + str(peaks_tollerance) + "signal_peaks_n: " + str(signal_peaks_n) + "scaled_savgol_filter_delta_peaks_n" + str(scaled_savgol_filter_delta_peaks_n))
            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_signal_vs_scaled_savgol_filter_delta_peaks_n_ratio'] = abs(signal_peaks_n - scaled_savgol_filter_delta_peaks_n) / period # peaks_n_ratio
            
                            
            if(time_tracking):
                self.track_time('\tPartial time Partial time Peaks cardinality Error calculation')   


            # -------------------------------------------------------------------------
            #         Correlation between current scaled harmonic and
            #         scaled delta derivate of long an short savgol_filtered signal
            # -------------------------------------------------------------------------
            averages['scaled_savgol_filter_delta_derivate'] = averages.diff()['scaled_savgol_filter_delta'] #averages['scaled_average_delta'].values - averages['scaled_average_delta'].shift(1).values #averages['scaled_average_delta'].diff()
            averages['scaled_savgol_filter_delta_derivate'] = averages['scaled_savgol_filter_delta_derivate'].fillna(0)
            signal['scaled_signal_derivate'] =  signal.diff()['scaled_signal'] #signal['scaled_signal'].values - signal['scaled_signal'].shift(1).values #signal['scaled_signal'].diff()
            signal['scaled_signal_derivate'] = signal['scaled_signal_derivate'] .fillna(0)

            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_savgol_filter_delta_derivate_correlations'] = simps(signal['scaled_signal_derivate'] * np.roll(averages['scaled_savgol_filter_delta_derivate'], tau), dx=1) / signal_peaks_n
                            
                            
            if(time_tracking):
                self.track_time('\tPartial time Partial time Correlation between scaled harmonic and ascled delta')   


            # ----------------------------------------------------------
            #         Peaks phase error between current scaled harmonic
            #         and scaled_savgol_filter_delta_correlation
            # -----------------------------------------------------------

            # Initialize a list to store the differences between the indices
            differences = []

            if(len(scaled_savgol_filter_delta_max_peaks_indexes) > 0 and len(signal_max_peaks_indexes) > 0):

                # find series with more max peaks
                if(len(scaled_savgol_filter_delta_max_peaks_indexes) >= len(signal_max_peaks_indexes)):
                    peak_indices_series1 = scaled_savgol_filter_delta_max_peaks_indexes
                    peak_indices_series2 = signal_max_peaks_indexes
                else:
                    peak_indices_series1 = signal_max_peaks_indexes
                    peak_indices_series2 = scaled_savgol_filter_delta_max_peaks_indexes

                # For each max peak in the first series, find the nearest peak in the second series
                for peak_index_series1 in peak_indices_series1:
                    differences_peak = np.abs(peak_index_series1 - peak_indices_series2)
                    nearest_peak_index = peak_indices_series2[np.argmin(differences_peak)]
                    differences.append(peak_index_series1 - nearest_peak_index)

            # find series with more min peaks
            if(len(scaled_savgol_filter_delta_min_peaks_indexes) > 0 and len(signal_min_peaks_indexes) > 0):

                # find series with more min peaks
                if(len(scaled_savgol_filter_delta_min_peaks_indexes) >= len(signal_min_peaks_indexes)):
                    peak_indices_series1 = scaled_savgol_filter_delta_min_peaks_indexes
                    peak_indices_series2 = signal_min_peaks_indexes
                else:
                    peak_indices_series1 = signal_min_peaks_indexes
                    peak_indices_series2 = scaled_savgol_filter_delta_min_peaks_indexes

                # For each min peak in the first series, find the nearest peak in the second series
                for peak_index_series1 in peak_indices_series1:
                    differences_peak = np.abs(peak_index_series1 - peak_indices_series2)
                    nearest_peak_index = peak_indices_series2[np.argmin(differences_peak)]
                    differences.append(peak_index_series1 - nearest_peak_index)

            if(len(differences) > 0):
            # Calculate the root mean square error of the differences
                root_mean_square_error = np.sqrt(np.mean(np.array(differences) ** 2))

            else:
                root_mean_square_error = period

            # Normalize with respect to the period length
            root_mean_square_error = root_mean_square_error / period

            goertzel_df_peaks.loc[goertzel_df_peaks['peaks_indexes'] == index, 'scaled_signal_vs_scaled_savgol_filter_delta_peaks_phase_RMSE'] = root_mean_square_error

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tPeaks analysis ennded, started global scoring')

                            
        if(time_tracking):
            self.track_time('\tPartial time error between current scaled harmonic and scaled_savgol_filter_delta_correlation')   

        # ------------------------------------------------------
        #         Calculate Cycles Global Scoring
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

        # print(goertzel_df_peaks.columns)
        if 'scaled_savgol_filter_delta_correlation' not in goertzel_df_peaks:
            print(goertzel_df_peaks)

        global_score = self.get_gloabl_score(goertzel_df_peaks, ascending_columns, descending_columns)

        goertzel_df_peaks['global_score'] = global_score

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tPeaksanalyss ended, started peaks sorting')
        
        if(time_tracking):
            self.track_time('\tPartial time Cicle Global Scoring calculation')   

        # ------------------------------------------------------
        #      Dominant peaks sorting
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
          # print("dominant_cicles_sorting_type == 'global_score'")

        else:
            sorted_indices = dominant_peaks['scaled_peak_amplitudes'].argsort()[::-1]
            sorted_dominant_peaks_indexes = dominant_peaks['dominant_peaks_indexes'].iloc[sorted_indices]
          # print("dominant_cicles_sorting_type == 'dominant_peaks_indexes'")

        used_indexes = sorted_dominant_peaks_indexes[0:final_kept_n_dominant_circles]


        # Final kept dominant cicles
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

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tpeaks sorting ended, started Dominant Circle Calibrated Standard Indicators')

        if(time_tracking):
            self.track_time('\tPartial time Dominants Peaks Sorting')   

        # ------------------------------------------------------
        #         Dominant Circle Calibrated Standard Indicators
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

        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tDominant Circle Calibrated Standard Indicators ended, started Averages Delta')
        
            
        if(time_tracking):
            self.track_time('\tPartial time Dominant Circle Calibrated Standard Indicator')   
            
        # ------------------------------------------------------
        #         Averages Delta
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
        #         scaled_savgol_filter_delta
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
        #         Indicators
        # ----------------------------------------------------------

        indicators_parameters = pd.DataFrame(columns = ['MACD_parameters', 'RSI_parameters', 'CAD_parameters'])
        
        if(indicators_signal_calcualtion == True):

            # if(data_column_name == 'RSI_column_name'):
            # print('\t\t\tDAverages Deltaended, started Averages indicators_signal_calcualtion')

            for index in used_indexes:

                phase = phases[index]
                amplitude = harmonics_amplitudes[index]
                frequency = frequency_range[index]
                period = 1 / frequency

                # ----------------------------------------------------------
                #         MACD and SAVGOL filter MACD (band pass filter)
                # ----------------------------------------------------------

                original_data, MACD_parameters = self.indict_MACD_SGMACD(data = original_data,
                                                                         signals_results = signals_results,
                                                                         dominant_period = period,
                                                                         data_column_name = data_column_name)


                # ----------------------------------------------------------
                #         RSI and SG smoothed RSI
                # ----------------------------------------------------------

                original_data, RSI_parameters = self.indict_RSI_SG_smooth_RSI(data = original_data,
                                                                              signals_results = signals_results,
                                                                              data_column_name = data_column_name,
                                                                              end_rebuilt_signal_index = end_rebuilt_signal_index,
                                                                              dominant_period = period)


                # ----------------------------------------------------------
                #         Centered Average Deltas
                # ----------------------------------------------------------

                original_data, CAD_parameters = self.indict_centered_average_deltas(data = original_data,
                                                                                     signals_results = signals_results,
                                                                                     data_column_name = data_column_name,
                                                                                     dominant_period = period)




                indicators_parameters = pd.concat([
                                                    indicators_parameters,
                                                    pd.DataFrame({
                                                        'MACD_parameters': [MACD_parameters],
                                                        'RSI_parameters': [RSI_parameters],
                                                        'CAD_parameters': [CAD_parameters]
                                                    })
                                                ], ignore_index=True)



            signals_results['indicators_parameters'] = []
            signals_results['indicators_parameters'] = indicators_parameters.apply(lambda row: row.to_dict(), axis=1)


        # if(data_column_name == 'RSI_column_name'):
        # print('\t\t\tDindicators_signal_calcualtion ended, started Dominant Circles Signal - singles and composite')
            
            
        if(time_tracking):
            self.track_time('\tPartial time Indicators (RSI, MACD, ...)') 
            
        # ----------------------------------------------------------
        #         Dominant Cicles Signal - single and composite
        # ----------------------------------------------------------

        time = np.linspace(0, num_samples*2, num_samples*2, endpoint=False)
        dominant_circle_signal = np.zeros(len(time), dtype=float)
        composite_dominant_cycle_signal = np.zeros(len(time), dtype=float)

        max_dominant_peak_period = 0
        max_dominant_peak_scaled_amplitude = 0
        max_period_dominant_circle = 0

#         signals_results['start_rebuilt_signal_index'] = [None] * len(signals_results)
        signals_results['start_rebuilt_signal_index'] = start_rebuilt_signal_index

#         signals_results['end_rebuilt_signal_index'] = [None] * len(signals_results)
        signals_results['end_rebuilt_signal_index'] = end_rebuilt_signal_index

        signals_results['dominant_peaks_signals'] = [None] * len(signals_results)
        
#         print(f"start_rebuilt_signal_index {start_rebuilt_signal_index}")
#         print(f"end_rebuilt_signal_index {end_rebuilt_signal_index}")
#         print(f"signals_results['dominant_peaks_signals'] {signals_results['dominant_peaks_signals']}")
#         print(f"len self.data {len(self.data)}")
#         print(f"len original_data {len(original_data)}")
#         print(f"len time {len(time)}")
        
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
                                               data = original_data,
                                               debug = False)      
            
#             print(f"\n\t- period {period}")
#             print(f"\t- extension_periods {extension_periods}")
#             print(f"\t- len signal {len(signal)}")


            if(extension_periods > 0):
                
                original_data = self.datetime_dateset_extend(original_data, extension_periods)


            # Single dominant sin cycle signals
#             original_data['dominant_circle_signal_period_' + str(period)] = signal
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
        
#         print(f"max_period_dominant_circle {max_period_dominant_circle}")

         # Composite dominant sin cycles signal
        signal, extension_periods = self.rebuilt_signal_zeros(signal = composite_dominant_cycle_signal,
                                           start_rebuilt_signal_index = start_rebuilt_signal_index,
                                           data = original_data,
                                           debug = False)
        
#         print(f"len composite signal {len(signal)}")
#         print(f"extension_periods {extension_periods}")
#         print(f"original data len {len(original_data)}")       


        if(extension_periods > 0):
#             print('extension_periods > 0')
            original_data = self.datetime_dateset_extend(original_data, extension_periods)
#         else:
#             print('extension_periods <= 0')

#         original_data['composite_dominant_circles_signal'] = signal
        new_columns['composite_dominant_circles_signal'] = signal
            
        if(time_tracking):
            self.track_time('\tPartial time Dominant Cicle Signals') 



        # ------------------------------------------------------
        #         Detrended Data
        # -----------------------------------------------------


#         original_data['detrended'] = detrended_data
        new_columns['detrended'] = detrended_data
    

        # Trova la lunghezza massima delle liste non vuote
        max_len = len(original_data)
        
#         print(f'MAX LEN {max_len}')


        # Riempie le colonne più corte con NaN
        for key in new_columns:

            col_len = len(new_columns[key])
            if col_len < max_len:

                add_len = max_len - col_len
        
                # Create a Series with NaNs to append
                new_nans = pd.Series([np.nan] * add_len, index=range(col_len, max_len))

                # Use pd.concat to append NaNs to the existing Series
                new_columns[key] = pd.concat([new_columns[key], new_nans])        

    
        new_data = pd.DataFrame(new_columns)        
        original_data = pd.concat([original_data, new_data], axis=1)
            
        if(time_tracking):
            self.track_time('\tPartial time Detrended Data') 


        # ------------------------------------------------------
        #         scaled_savgol_filter_delta
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
        #         Print Data results
        # ------------------------------------------------------

        if(print_report == True):
#                 print('ANALYZED SAMPLES NUMBER: ' + str(len(data)))
#                 config_data = [(key, value) for key, value in configuration.items()]
#                 table = tabulate(config_data, headers=["Parameter", "Value"], tablefmt="fancy_grid")
#                 print(table)
            display(configuration)



            print("\n")
#                 print_data = goertzel_df_peaks.values.tolist()
#                 print_headers = goertzel_df_peaks.columns.tolist()
#                 table = tabulate(print_data, print_headers, tablefmt="fancy_grid")
#                 print(table)
            display(goertzel_df_peaks)

            print("\n")
#                 print_data = kept_dominant_peaks.values.tolist()
#                 print_headers = kept_dominant_peaks.columns.tolist()
#                 table = tabulate(print_data, print_headers, tablefmt="fancy_grid")
#                 print(table)
            display(kept_dominant_peaks)



        # ------------------------------------------------------
        #         Charting
        # ------------------------------------------------------


        if(show_charts == True):

            # Parameters
            basic_rows_n = 5

            if(include_calibrated_MACD == True):
                basic_rows_n +=1
                MACD_row_number = basic_rows_n


            if(include_calibrated_RSI== True):
                basic_rows_n +=1
                RSI_row_number = basic_rows_n


            # Generalized Goertzel Transform Spectrum Chart
            spectrum_trace = go.Scatter(x=frequency_range, y=harmonics_amplitudes, mode='lines', name='Goetzel DFT Spectrum')
            fig_spectrum = go.Figure(spectrum_trace)
            fig_spectrum.update_layout(title="Frequency Spectrum", xaxis=dict(title="Frequency"), yaxis=dict(title="Magnitude"))

            fig_spectrum.show()


            # Original data, detrended, dominant circles signal, averages delta
            fig = make_subplots(rows=basic_rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Original Data", "Detrended Data", "Dominant Circles Signal", "Centered Averages Delta"))


            fig.add_trace(go.Scatter(x=original_data.index,
                                      y=original_data[data_column_name],
                                      mode="lines",
                                      name="Original data"),
                        row=1,
                        col=1)

            # Add a vertical line spanning the height of the chart
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=original_data[data_column_name].min(),
                y1=original_data[data_column_name].max(),
                line=dict(color='purple', width=1),
                row=1, col=1
            )


            fig.add_trace(go.Scatter(x=original_data.index, y=original_data['detrended'] , mode="lines", name="Detrended Close"), row=2, col=1)
            fig.add_trace(go.Scatter(x=original_data.index, y=original_data['composite_dominant_circles_signal'] , mode="lines", name="Dominant Circle Signal"), row=3, col=1)



            # Add a vertical line spanning the height of the chart
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=original_data['composite_dominant_circles_signal'].min(),
                y1=original_data['composite_dominant_circles_signal'].max(),
                line=dict(color='purple', width=1),
                row=3, col=1
            )

            fig.add_trace(go.Scatter(x=original_data.index, y=original_data['centered_averages_delta']  , mode="lines", name="Centered Averages Delta"), row=4, col=1)


            fig.add_trace(go.Scatter(x=original_data.index, y=original_data['scaled_savgol_filter_delta']  , mode="lines", name="Centered Averages Delta"), row=5, col=1)


            if(include_calibrated_MACD == True):

                for i in range(len(signals_results)):

                    col_name = signals_results.iloc[i]['indicators_parameters']['MACD_parameters']['savgol_MACD_signal_name']

                    fig.add_trace(go.Scatter(x=original_data.index, y=original_data[col_name], mode="lines", name=col_name), row=MACD_row_number, col=1)




            if(include_calibrated_RSI== True):

                for i in range(len(signals_results)):
                    col_name = signals_results.iloc[i]['indicators_parameters']['RSI_parameters']['smoothed_RSI_name']

                    fig.add_trace(go.Scatter(x=original_data.index, y=original_data[col_name], mode="lines", name=col_name), row=RSI_row_number, col=1)



            fig.update_layout(title="Goertzel Dominant Circles Analysis", height=2000)


            fig.update_xaxes(type="category")

            # Visualizza il secondo grafico con i subplot
            fig.show()



        return current_date, index_of_max_time_for_cd, original_data, signals_results, configuration


    def multiperiod_analysis(self,
                             data_column_name,
                             current_date,
                             periods_pars,
                             best_fit_start_back_period = None,                             
                             time_zone = 'America/New_York',
                             pars_from_opt_file = False,
                             files_path_name = None,
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
                             bb_delta_fixed_periods = None,
                             bb_delta_sg_filter_window = None,
                             RSI_cycles_analysis_type = 'original_RSI', # 'SG_smooted_RSI',
                             enable_cycles_alignment_analysis = True,
                             opt_algo_type = 'genetic_omny_frequencies',  # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency'
                             detrend_type = 'hp_filter',
                             lowess_k = 3,
                             linear_filter_window_size_multiplier = 1,
                             period_related_rebuild_range = False,
                             period_related_rebuild_multiplier = 2.5,
                             CDC_bb_analysis = True,
                             CDC_RSI_analysis = True,
                             CDC_MACD_analysis = True,
                             enabled_multiprocessing = True,
                             time_tracking = False
                            ):

        
        print(f'Data column name: {data_column_name}')
        
        scaler = self.scaler


        self.MultiAn_fitness_type = MultiAn_fitness_type
        self.best_fit_start_back_period = best_fit_start_back_period
        self.period_related_rebuild_range = period_related_rebuild_range
        self.period_related_rebuild_multiplier = period_related_rebuild_multiplier

        elaborated_data_series = [] # pd.DataFrame()
        signals_results_series = [] # pd.DataFrame()
        configurations_series = []
        goertzel_amplitudes = []
        
        if(detrend_type != 'linear' and detrend_type != 'lowess'):
            detrend_type = 'hp_filter'


        if(self.print_activity_remarks):            
            print('\nMultiperiod analysis')            
            self.track_time('1. multiperiod_analysis: entering for loop, start calling analyze_and_plot')
        
        for index, row in periods_pars.iterrows():

            num_samples = row['num_samples']
            final_kept_n_dominant_circles = row['final_kept_n_dominant_circles']
            min_period = row['min_period']
            max_period = row['max_period']
            hp_filter_lambda = row['hp_filter_lambda']
            
            if('detrend_type' in row.index and row['detrend_type'] is not None):
                detrend_type = row['detrend_type']
            else:
                detrend_type = 'hp_filter'

            
            hp_filter_lambda = 1600 # not null default value
            lowess_k = 3 # not null default value
            
            if(detrend_type == 'hp_filter'):
                hp_filter_lambda = row['hp_filter_lambda']
                
            if(detrend_type == 'lowess'):
                lowess_k = row['lowess_k']
                
            display(row)
            
            print(f'\nStarted periods analysis in range[{min_period}, {max_period}]')
            print(f'\tNum samples {num_samples}')
            print(f'\tdetrend_type {detrend_type}')
            print(f'\tlowess_k {lowess_k}')

            max_length_series = 0
            max_length_series_index = -1
            


            _, index_of_max_time_for_cd, elaborated_data, signal_results, configuration = self.analyze_and_plot(
                             data = self.data,
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
                             centered_averages = True,
                             time_zone = time_zone,
                             other_correlations = True,
                             include_calibrated_MACD = True,
                             include_calibrated_RSI = True,
                             show_charts = False,
                             print_report = False,
                             time_tracking = time_tracking
                            )
        
            
            elaborated_data_series.append(elaborated_data) # pd.concat([elaborated_data_df, elaborated_data], ignore_index=True)
            signals_results_series.append(signal_results) # pd.concat([signals_results_df, signal_results], ignore_index=True)
            configurations_series.append(configuration)

            print(f"index {index}")
            print(f"min_period {min_period}")
            if(len(elaborated_data) > max_length_series):

                max_length_series = len(elaborated_data)
                max_length_series_index = index


        # ---------------------------------------------------------------
        #       Re-Factorize Cyrcles Amplitude
        # ---------------------------------------------------------------
        if(self.print_activity_remarks):
            print('\nRe-Factorization of Cyrcles Amplitude')        
            self.track_time('\n2. multiperiod_analysis: starting Re-Factorize Cyrcles Amplitude')
# qui qui
        # find index of analysis with higest HP lambda value, keep its detrended signal
        if(reference_detrended_data == "less_detrended"):
            if(detrend_type == 'hp_filter'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['hp_filter_lambda'])
            if(detrend_type == 'lowess'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: (configurations_series[i]['lowess_k'] * configurations_series[i]['max_period']))

        # find index of analysis with highest number of considered samples, keep its detrended signal
        if(reference_detrended_data == "longest"):
            index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['num_samples'])
            
        print(f'index_detrended_data {index_detrended_data}')

        self.MultiAn_reference_detrended_data = elaborated_data_series[index_detrended_data]['detrended'][0:index_of_max_time_for_cd+1]

        if(MultiAn_fitness_type_svg_smoothed == True):
            self.MultiAn_reference_detrended_data = savgol_filter(self.MultiAn_reference_detrended_data, MultiAn_fitness_type_svg_filter, 2)

        self.MultiAn_reference_detrended_data = scaler.fit_transform( self.MultiAn_reference_detrended_data.values.reshape(-1, 1)).flatten() * 100


        # find min and max values of detrended signal
        self.MultiAn_detrended_max = np.int64(self.MultiAn_reference_detrended_data.max())
        self.MultiAn_detrended_min = np.int64(self.MultiAn_reference_detrended_data.min())


        # create structure with meaningful data for each cycle
#         self.MultiAn_dominant_cycles_df = pd.DataFrame(columns=['peak_frequencies',
#                                                                 'peak_periods',
#                                                                 'peak_phases',
#                                                                 'start_rebuilt_signal_index',
#                                                                 'end_rebuilt_signal_index'])
        
        # RESET for new analysis to skip values of past ones
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

#                 display(structure)
                peak_frequencies = structure['peak_frequencies']
                peak_periods = structure['peak_periods']
                peak_phases = structure['peak_phases']
                start_rebuilt_signal_index = int(structure['start_rebuilt_signal_index'])
                end_rebuilt_signal_index = int(structure['end_rebuilt_signal_index'])

                if(max_start_rebuilt_signal_index < start_rebuilt_signal_index):
                    max_start_rebuilt_signal_index = start_rebuilt_signal_index

                df_row = pd.DataFrame({
                    'peak_frequencies': [peak_frequencies],
                    'peak_periods': [peak_periods],
                    'peak_phases': [peak_phases],
                    'start_rebuilt_signal_index': [start_rebuilt_signal_index],
                    'end_rebuilt_signal_index': [end_rebuilt_signal_index]
                })


                self.MultiAn_dominant_cycles_df = pd.concat([self.MultiAn_dominant_cycles_df, df_row], ignore_index=True)
                
#                 # Convert the columns to int64
#                 self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'] = self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].astype('int64')
#                 self.MultiAn_dominant_cycles_df['end_rebuilt_signal_index'] = self.MultiAn_dominant_cycles_df['end_rebuilt_signal_index'].astype('int64')

                
#                 print(f"type self.MultiAn_dominant_cycles_df.iloc[0]['start_rebuilt_signal_index'] : {type(self.MultiAn_dominant_cycles_df.iloc[0]['start_rebuilt_signal_index'] )}")
                
#                 print('\n\n-------------------')
#                 print('self.MultiAn_dominant_cycles_df')
#                 print(self.MultiAn_dominant_cycles_df)
#                 print('-------------------')

#                 # Trova e stampa le righe con NaN in self.MultiAn_dominant_cycles_df
#                 nan_rows_multiAn = self.MultiAn_dominant_cycles_df[self.MultiAn_dominant_cycles_df.isnull().any(axis=1)]
#                 if not nan_rows_multiAn.empty:
#                     print('\n\n--------------------------------------------')
#                     print("Righe con NaN in self.MultiAn_dominant_cycles_df:")
#                     print(nan_rows_multiAn)
#                     print('\n\n--------------------------------------------')
                    
#                 print('\n\n-------------------')
#                 print('df_row')
#                 print(df_row)
#                 print('-------------------')

#                 # Trova e stampa le righe con NaN in df_row
#                 nan_rows_df_row = df_row[df_row.isnull().any(axis=1)]
#                 if not nan_rows_multiAn.empty:
#                     print('\n\n--------------------------------------------')
#                     print("Righe con NaN in df_row:")
#                     print(nan_rows_df_row)
#                     print('\n\n--------------------------------------------')
          
            

        # descending order to simplify possible segmented best amplitudes search
        self.MultiAn_dominant_cycles_df = self.MultiAn_dominant_cycles_df.sort_values(by='peak_periods', ascending=False)
#         print('\nDESCENDING ORDERED DOMINANT CYCLES:')
#         print(self.MultiAn_dominant_cycles_df)



        cycles_n = len(self.MultiAn_dominant_cycles_df )
        detrended_abs_max = self.MultiAn_detrended_max -self.MultiAn_detrended_min
        up_series = pd.Series([detrended_abs_max] * cycles_n)
        low_series = pd.Series([0] * cycles_n)
        
        if(self.print_activity_remarks):
            self.track_time('3. multiperiod_analysis: end Re-Factorize Cyrcles Amplitude')
        
        

        # -------------------------------------------------------
        # USE GOERTZEL FOR FINDING BEST AMPLITUDES
        # -------------------------------------------------------
        
        if(self.print_activity_remarks):
            self.track_time('\n4. multiperiod_analysis: starting Goertzel best amplitudes')

#         goertzel_best_refactoring_df = pd.DataFrame(columns = ['peak_periods', 'peak_phases', 'peak_amplitudes'])
        goertzel_best_refactoring_df = pd.DataFrame({
            'peak_periods': pd.Series(dtype='float64'),
            'peak_frequencies': pd.Series(dtype='float64'),
            'peak_phases': pd.Series(dtype='float64'),
            'peak_amplitudes': pd.Series(dtype='float64')
        })
        new_rows = []
        
#         print('self.MultiAn_dominant_cycles_df:')
#         display(self.MultiAn_dominant_cycles_df)

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
        
        if(self.print_activity_remarks):
            self.track_time('\n5. multiperiod_analysis: end Goertzel best amplitudes')
        
        
        
        # -------------------------------------------------------------------
        # OPTIMIZE SINGLE FREQUENCY IN ADDITION STARTING FROM SLOWER ONES
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

           
            
            # loop on periods descending ordered
            for index, row in self.MultiAn_dominant_cycles_df.iterrows():
                
                comparision_length = int(2.5 * row['peak_periods']) # comparision length
                length = (len_series - row['start_rebuilt_signal_index'])
                start_comparison_index = len_series - comparision_length
                
                # loop on possible amplitudes descending order
                for temp_amp in np.arange(self.MultiAn_detrended_max, 0, -0.01):                    
                    
                    
                    temp_circle_signal = pd.Series([0.0] * len_series)
                    time = np.linspace(0, length, length, endpoint=False)

                    temp_circle_signal[row['start_rebuilt_signal_index']:] = temp_amp * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
                    # 20241119: possible error, replace composite_dominant_cycle_signal with zeros series?
                    temp_rebuilt_signal = composite_dominant_cycle_signal + temp_circle_signal
                    
                    error = mean_squared_error(self.MultiAn_reference_detrended_data[start_comparison_index:], temp_rebuilt_signal[start_comparison_index:]) 
                    
                   
                    if(error < last_error):                        
                        best_amplitude = temp_amp 
                        best_error = error
#                         print('\t\tPeriod ' + str(row['peak_periods']) + f' found better amplitude {best_amplitude}, mse = {error}')
                        
                    last_error = error
                    
                    
                # add the best amplitude for this period on the reuilding domainant cycles signal
                amplitudes.append(best_amplitude)
                best_fitness_value = best_error
                print('Period ' + str(row['peak_periods']) + f', best amplitude {best_amplitude}, best_fitness_value {best_fitness_value}')
                
                # add the last cycle with best ampliude to the composite signal
                composite_dominant_cycle_signal = composite_dominant_cycle_signal + temp_rebuilt_signal
           
            if(self.print_activity_remarks):    
                print('Single cycle best fitting, aplitudes:')
                print(amplitudes)
            
           
                


        # -------------------------------------------------------
        # USE GENETICS ALGORITHM FOR FINDING BEST AMPLITUDES
        # -------------------------------------------------------
        # # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency'
        
        elif(opt_algo_type == 'genetic_omny_frequencies'):
            
 
            self.track_time('\n6. Genetics algo for best amplitudes identification, start inizializing')
    
            # Crea il creator per il tipo di fitness (massimizzazione di var1 e minimizzazione di var2 e var3)
            if 'FitnessMulti' not in creator.__dict__:
                creator.create("FitnessMulti", base.Fitness, weights=(weigth,)) # loss are always negative so they must be maximized

            # Crea il creator per un individuo (una lista di parametri)
            if 'Individual' not in creator.__dict__:
                creator.create("Individual", list, fitness=creator.FitnessMulti)

            # Inizializzazione della popolazione e delle operazioni genetiche
            toolbox = base.Toolbox()
            toolbox.register("individual", tools.initIterate, creator.Individual, self.MultiAn_initializeIndividual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self.MultiAn_evaluateFitness)
            toolbox.register("mate", self.custom_crossover) # tools.cxTwoPoint)
            toolbox.register("mutate",
                          tools.mutUniformInt,
                          low = low_series.to_list(), #[0, 0, 0, 0, 0, 0, 0],
                          up = up_series.to_list(), #[256, 256, 256, 256, 256, 256, 256],
                          indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            if( enabled_multiprocessing == True):
                
                # Set the multiprocessing start method to 'spawn'
                if multiprocessing.get_start_method() != 'spawn':
                    multiprocessing.set_start_method('spawn')

                cpu_count = multiprocessing.cpu_count()
                print(f"CPU count: {cpu_count}")

                pool = multiprocessing.Pool()
                toolbox.register("map", pool.map)


            # Create the initial population
            population = toolbox.population(n=population_n)
            
            self.track_time('6. Genetics algo for best amplitudes identification, end inizialization')
        

            for gen in range(NGEN):
                
                self.track_time('\n7. Genetics algo in generations loop, start new loop')

                offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
                fits = toolbox.map(toolbox.evaluate, offspring)
                count = 0
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                    count += 1
                    
#                     clear_output(wait=True)
#                     print("Generation number: " + str(gen+1) + "/" + str(NGEN)+ ", population number " + str(count) + "/" + str(population_n))

#                     sys.stdout.flush()
                    
                self.track_time('8. Genetics algo in generations loop, loop ended')
                    

                population = toolbox.select(offspring, k=len(population))

                best_individual = tools.selBest(population, k=1)[0]
                best_fitness = best_individual.fitness.values


            if(enabled_multiprocessing == True):
                pool.close()
                pool.join()

            best_individual = tools.selBest(population, k=1)[0]
            best_fitness = best_individual.fitness.values

#             clear_output(wait=True)

            if(self.print_activity_remarks):
                print("\n\n--------------------------------------------------------")
                print("Multirange Analysis Genetics Optimization results:")
                print('\tbest_individual: ' + str(best_individual))
                print('\tbest_fitness: ' + str(best_fitness))
                print("--------------------------------------------------------")

            self.MultiAn_dominant_cycles_df['best_amplitudes'] = best_individual

            best_fitness_value = best_fitness[0]
            self.MultiAn_dominant_cycles_df['best_fitness'] = best_fitness_value

            display(self.MultiAn_dominant_cycles_df)

            amplitudes = best_individual


        # -------------------------------------------------------
        # USE TPE ALGORITHMS FOR FINDING BEST AMPLITUDES
        # -------------------------------------------------------

        elif(opt_algo_type ==  'tpe' or opt_algo_type ==  'atpe'):

            # Funzione obiettivo da massimizzare
            def objective(params):
                amplitude_values = [params[f'amplitude_{i}'] for i in range(len(low_series))]
                # Chiamata alla tua funzione di valutazione fitness
                fitness = self.MultiAn_evaluateFitness(amplitude_values, False)
                return fitness  # Minimizzazione, quindi neghiamo il valore di fitness

            # Definizione dello spazio dei parametri da ottimizzare
            space = {f'amplitude_{i}': hp.uniform(f'amplitude_{i}', low_series[i], up_series[i]) for i in range(len(low_series))}

            # Ottimizzazione con TPE
            if(opt_algo_type ==  'tpe'):
                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=NGEN)

            if(opt_algo_type ==  'atpe'):
                best = fmin(fn=objective, space=space, algo=atpe.suggest, max_evals=NGEN)


            # Estrai i migliori valori degli amplitudes
            best_amplitudes = [best[f'amplitude_{i}'] for i in range(len(low_series))]

            # Visualizza i risultati
            print("Best Amplitudes:", best_amplitudes)

            # Aggiorna la tua struttura dati con i risultati ottenuti
            self.MultiAn_dominant_cycles_df['best_amplitudes'] = best_amplitudes

            # Calcola il miglior fitness e aggiorna la struttura dati
            best_fitness_value = objective(best)
            self.MultiAn_dominant_cycles_df['best_fitness'] = best_fitness_value

            # Visualizza i risultati finali
#             print("Final results")
#             display(self.MultiAn_dominant_cycles_df)

            amplitudes = best_amplitudes

        else:

            print("Errore: optmization type not in list ('mono_frequency', 'genetic_omny_frequencies', 'tpe', 'atpe')")
            return None, None, None, None, None, None

#         print("\n\nGoertzel refactoring results: ")
#         display(goertzel_best_refactoring_df)

        # -----------------------------------------------------------------------
        #                       CREATE COMPOSITE SIGNALS
        # -----------------------------------------------------------------------

        if(self.print_activity_remarks):
#             print('\nComposite signals creation')
            self.track_time('\n9. Genetics start composite signal creation')


        temp_circle_signal = []
        composite_dominant_cycle_signal = []

        # the portion of new wave will be long twice the longest considered portion of detrended signal
        # to cover it and an equal length of future projection
        len_series = len(self.data) #len(self.MultiAn_reference_detrended_data)

        # the global length of the series containing the dominant cycles will be long as the original data
        # plus the projection. The Projection will be long as much as the portion of elaboration of the
        # longest start_rebuilt_signal_index
        print(f"len_series {len_series}")
        print(f"len elaborated_data_series {len(elaborated_data_series)}")
        print(f"max_length_series_index {max_length_series_index}")

        
        df_indexes_list = elaborated_data_series[max_length_series_index].index

        self.track_time('\t9.a. Genetics start composite_signal')
        composite_signal = self.cicles_composite_signals(max_length_series, amplitudes,self.MultiAn_dominant_cycles_df, df_indexes_list, 'composite_signal')       


        self.track_time('\t9.b. Genetics: end composite_signal, start alignmentsKPI')
        
        alignmentsKPI = pd.Series()
        weigthed_alignmentsKPI = pd.Series()
        
        if(enable_cycles_alignment_analysis == True):
                                        
            alignmentsKPI, weigthed_alignmentsKPI = self.MultiAn_cyclesAlignKPI(composite_signal.drop(['composite_signal'], axis=1),
                                                                            max_start_rebuilt_signal_index,
                                                                            amplitudes,
                                                                            self.MultiAn_dominant_cycles_df['peak_periods'])
            
        else:
            if(self.print_activity_remarks):
                print('alignmentsKPI and weigthed_alignmentsKPI analysis disabled')

        if(self.print_activity_remarks):
            self.track_time('\t9.c. Genetics: end alignmentsKPI, start cicles_composite_signals')    

        temp = self.cicles_composite_signals(max_length_series, goertzel_amplitudes, goertzel_best_refactoring_df, df_indexes_list, 'goertzel_composite_signal')
        
        composite_signal = pd.concat([composite_signal, temp], axis=1)

        if(self.print_activity_remarks):
            self.track_time('10. Genetics end composite and alignmentsKPI signal creation')


        # ----------------------------------------------------------
        # BB DELTA AS KPI OF VOLATILITY, FOR EACH IDENTIFIED PERIOD
        # AND CYCLES ANALYSIS ON IT AND ITS TWO DERIVATES
        # ----------------------------------------------------------
        
        max_start_index = 0
        start_index = end_rebuilt_signal_index - num_samples

        if(start_index > max_start_index):
            max_start_index = start_index
            
        bb_delta = pd.DataFrame(index=elaborated_data_series[max_length_series_index].index)

        if(CDC_bb_analysis == True):
            
            if(self.print_activity_remarks):
                print('\nBollinger Bands Volatility Cycles Analysis')

#             bb_delta = pd.DataFrame(index=elaborated_data_series[max_length_series_index].index)
            bb_delta_len = len(bb_delta)
            temp_upper_band = []
            temp_middle_band = []
            temp_lower_band = []

            if(bb_delta_fixed_periods == None):

                BB_periods = []

                for index, row in self.MultiAn_dominant_cycles_df.iterrows():
                    BB_periods.append(1/row['peak_frequencies'])

            else:

                BB_periods = bb_delta_fixed_periods


#             max_start_index = 0
            multiple_factor = 2.5

            for temp_period in BB_periods:

                if(self.print_activity_remarks):
                    print('\tBB period: ' + str(temp_period))

                temp_upper_band, temp_middle_band, temp_lower_band = talib.BBANDS(self.data[data_column_name], timeperiod = temp_period)

                if(bb_delta_sg_filter_window is not None):

                    bb_delta['BB_delta_' + str(temp_period)] = temp_upper_band - temp_lower_band
                    bb_delta['BB_delta_' + str(temp_period)] = savgol_filter(bb_delta['BB_delta_' + str(temp_period)], bb_delta_sg_filter_window, 2)

                    bb_delta['BB_delta_der^1_' + str(temp_period)] = bb_delta['BB_delta_' + str(temp_period)].diff().fillna(0)
                    bb_delta['BB_delta_der^1_' + str(temp_period)] = savgol_filter(bb_delta['BB_delta_der^1_' + str(temp_period)], bb_delta_sg_filter_window, 2)


                    bb_delta['BB_delta_der^2_' + str(temp_period)] = bb_delta['BB_delta_der^1_' + str(temp_period)] .diff().fillna(0)
                    bb_delta['BB_delta_der^2_' + str(temp_period)] = savgol_filter(bb_delta['BB_delta_der^2_' + str(temp_period)], bb_delta_sg_filter_window, 2)

                else:
                    bb_delta['BB_delta_' + str(temp_period)] = temp_upper_band - temp_lower_band
                    bb_delta['BB_delta_der^1_' + str(temp_period)] = bb_delta['BB_delta_' + str(temp_period)].diff().fillna(0)
                    bb_delta['BB_delta_der^2_' + str(temp_period)] = bb_delta['BB_delta_der^1_' + str(temp_period)] .diff().fillna(0)


                temp_bb_delta = bb_delta.iloc[0:end_rebuilt_signal_index+1]
#                 temp_bb_delta = bb_delta
                temp_bb_delta = temp_bb_delta.fillna(0)
        
                print(f'len temp_bb_delta {len(temp_bb_delta)}')

                min_period = 6
                max_period = int(temp_period * multiple_factor)
                num_samples = max_period * 4

                if(temp_period <= 8):
                    hp_filter_lambda = 40
                elif(temp_period <= 16):
                    hp_filter_lambda = 80
                elif(temp_period <= 32):
                    hp_filter_lambda = 1021
                elif(temp_period <= 64):
                    hp_filter_lambda = 21964                
                elif(temp_period <= 128):
                    hp_filter_lambda = 7303660
                elif(temp_period <= 256):
                    hp_filter_lambda = 212103727
                elif(temp_period > 256):
                    hp_filter_lambda = 2000000000


                if(CDC_bb_analysis == True):

                        clear_output(wait=True)
                        print('-----------------------------')
                        print('      CDC ANALYSIS FOR BB')
                        print('-----------------------------')
                        
                        # Cycles analysis for BB
                        _, _, temp_elaborated_data, _, _ =  self.analyze_and_plot(
                                    data =  temp_bb_delta,
                                    data_column_name = 'BB_delta_' + str(temp_period),
                                    transform_precision = 0.01,
                                    num_samples = num_samples,
                                    start_date = None,
                                    current_date = current_date,
                                    final_kept_n_dominant_circles = 7,
                                    dominant_cicles_sorting_type = 'global_score',
                                    limit_n_harmonics = None,
                                    min_period = min_period,
                                    max_period = max_period,
                                    detrend_type = 'hp_filter',
                                    lowess_k = lowess_k,
                                    bartel_peaks_filtering = True,
                                    bartel_scoring_threshold = 0,
                                    hp_filter_lambda = hp_filter_lambda, #10**8 * temp_period ,
                                    jp_filter_p = 4,
                                    jp_filter_h = 8,
                                    centered_averages = True,
                                    time_zone = 'America/New_York',
                                    other_correlations = True,
                                    include_calibrated_MACD = False,
                                    include_calibrated_RSI = False,
                                    show_charts = False,
                                    print_report = False,
                                    indicators_signal_calcualtion = False, 
                                    time_tracking = time_tracking)

                        if(temp_elaborated_data is not None):

                            bb_min = min(bb_delta['BB_delta_' + str(temp_period)][start_index:end_rebuilt_signal_index+1])
                            bb_max = max(bb_delta['BB_delta_' + str(temp_period)][start_index:end_rebuilt_signal_index+1])

                            len_bb_CDC = len(temp_elaborated_data['composite_dominant_circles_signal'])
                            
#                             temp_elaborated_data['composite_dominant_circles_signal'].plot()

                            if(len_bb_CDC < bb_delta_len):
                                end_index = len_bb_CDC - 1

                            else:
                                end_index = bb_delta_len - 1

                            bb_scaler = MinMaxScaler(feature_range=(bb_min, bb_max))

                            # crea serie di lunghezza len(bb_delta) riempita di Nan
#                             bb_delta['BB_delta_CDC_' + str(temp_period)] = pd.Series([float('nan')] * len(bb_delta))
                            bb_delta['BB_delta_CDC_' + str(temp_period)] = pd.Series([float('nan')] * (end_index + 1))
                            
#                             print(f'BB delta CDC')
                            print(f"len self.data[data_column_name] {len(self.data[data_column_name])}")
                            print(f"len composite_signal {len(composite_signal)}")
                            print(f'bb_delta len {len(bb_delta)}')
                            print(f'temp_elaborated_data len {len(temp_elaborated_data)}')
                            print(f'\t- start_index: {start_index}')
                            print(f"\t- end_rebuilt_signal_index: {end_rebuilt_signal_index}")
                            print(f'\t- end_index: {end_index}')
                            print(f'\t- len_bb_CDC: {len_bb_CDC}')
                            print(f'\t- bb_delta_len: {bb_delta_len}')
                            
                            print('PRIMA')
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+1] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+1]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+2] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+2]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index-1] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index-1]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index]}")
                            # riempi serie con valori non Nan (sicuri?) tra start_index e end_index
#                             bb_delta.iloc[start_index:end_index, bb_delta.columns.get_loc('BB_delta_CDC_' + str(temp_period))] = bb_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:end_index].values.reshape(-1, 1)).flatten() 
#                             bb_delta.iloc[start_index:, bb_delta.columns.get_loc('BB_delta_CDC_' + str(temp_period))] = bb_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:].values.reshape(-1, 1)).flatten() 
                            bb_delta.iloc[start_index:end_index, bb_delta.columns.get_loc('BB_delta_CDC_' + str(temp_period))] = bb_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'].values.reshape(-1, 1)).flatten()[start_index:end_index]
                            print('DOPO')
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+1] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+1]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+2] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_rebuilt_signal_index+2]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index-1] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index-1]}")
                            print(f"b_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index] {bb_delta['BB_delta_CDC_' + str(temp_period)].iloc[end_index]}")
        


#                             print(f'start date bb_delta {bb_delta.iloc[start_index]}')
#                             print(f'start date bb_delta {bb_delta.iloc[end_index]}')
# #                             print(f"start date temp_elaborated_data {temp_elaborated_data['composite_dominant_circles_signal'].iloc[start_index]}")
# #                             print(f"start date temp_elaborated_data {temp_elaborated_data['composite_dominant_circles_signal'].iloc[end_index]}")
                                                        
                            
#                             display(bb_delta['BB_delta_CDC_' + str(temp_period)])
                            
#                             print(f"len temp_elaborated_data['composite_dominant_circles_signal'] {len(temp_elaborated_data['composite_dominant_circles_signal'])}")
                            
#                             print("    temp_elaborated_data['composite_dominant_circles_signal'] ")
#                             display(temp_elaborated_data['composite_dominant_circles_signal'])
#                             print("temp_bb_delta")
#                             display(temp_bb_delta)
                            
                        else:
                            if(self.print_activity_remarks):
                                print('Error in cycles analysis for bb_delta')
                                print('len temp_bb_delta: ' + str(len(temp_bb_delta)))


                        # Cycles analysis for BB^1
                        _, _, temp_elaborated_data, _, _ =  self.analyze_and_plot(
                                    data =  temp_bb_delta,
                                    data_column_name = 'BB_delta_der^1_' + str(temp_period),
                                    transform_precision = 0.01,
                                    num_samples = num_samples,
                                    start_date = None,
                                    current_date = current_date,
                                    final_kept_n_dominant_circles = 7,
                                    dominant_cicles_sorting_type = 'global_score',
                                    limit_n_harmonics = None,
                                    min_period = min_period,
                                    max_period = max_period,
                                    detrend_type = 'hp_filter',
                                    lowess_k = lowess_k,
                                    bartel_peaks_filtering = True,
                                    bartel_scoring_threshold = 0,
                                    hp_filter_lambda = hp_filter_lambda, #1000 * 10 ** ( temp_period * multiple_factor / 20 ), # 10**8 * temp_period ,
                                    jp_filter_p = 4,
                                    jp_filter_h = 8,
                                    centered_averages = True,
                                    time_zone = 'America/New_York',
                                    other_correlations = True,
                                    include_calibrated_MACD = False,
                                    include_calibrated_RSI = False,
                                    show_charts = False,
                                    print_report = False,
                                    indicators_signal_calcualtion = False,
                                    time_tracking = time_tracking)

                        if(temp_elaborated_data is not None):

                            bb_min = min(bb_delta['BB_delta_der^1_' + str(temp_period)][start_index:end_rebuilt_signal_index+1])
                            bb_max = max(bb_delta['BB_delta_der^1_' + str(temp_period)][start_index:end_rebuilt_signal_index+1])

                            len_bb_CDC = len(temp_elaborated_data['composite_dominant_circles_signal'])

                            if(len_bb_CDC < bb_delta_len):
                                end_index = len_bb_CDC

                            else:
                                end_index = bb_delta_len

                            bb_scaler = MinMaxScaler(feature_range=(bb_min, bb_max))

                            # print('\nlen(bb_delta): ' + str(len(bb_delta)))
                            # print('start_index: ' + str(start_index))
                            # print('len_bb_CDC: ' + str(len_bb_CDC))

                            bb_delta['BB_delta_CDC_der^1_' + str(temp_period)] = pd.Series([float('nan')] * len(bb_delta))
                            bb_delta.iloc[ start_index:end_index, bb_delta.columns.get_loc('BB_delta_CDC_der^1_' + str(temp_period)) ] = bb_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:end_index].values.reshape(-1, 1)).flatten()#['BB_delta_CDC_der^1_' + str(temp_period)].iloc[start_index:end_index] = bb_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:end_index].values.reshape(-1, 1)).flatten()

    # df["col"][row_indexer] = value

    # Use `df.loc[row_indexer, "col"] = values`

                        else:
                            if(self.print_activity_remarks):
                                print('Error in cycles analysis for bb_delta')
                                print('len temp_bb_delta: ' + str(len(temp_bb_delta)))



        # -----------------------------------------------------------------------
        #                       CYCLES ANALYSIS FOR RSI
        # -----------------------------------------------------------------------

        cdc_rsi = pd.DataFrame(index=elaborated_data_series[max_length_series_index].index)
        if(CDC_RSI_analysis == True):

            if(self.print_activity_remarks):
                print('\nCYCLES ANALYSIS FOR RSI')

            max_series_len = len(elaborated_data_series[max_length_series_index])
            count = 0
            for signal_serie in signals_results_series:

                for indicator_parameters in signal_serie['indicators_parameters']:

                    temp_RSI_name = indicator_parameters['RSI_parameters']['RSI_name']
                    temp_smooted_RSI_name = indicator_parameters['RSI_parameters']['smoothed_RSI_name']
                    temp_indicator_period = indicator_parameters['RSI_parameters']['indicator_period']



                    if(temp_indicator_period >= 6):

                        print('\tPeriod: ' + str(temp_indicator_period))

                        # elaborated_data_series[count][temp_RSI_name],

                        # temp_len = len(elaborated_data_series)
                        # print("elaborated_data_series[count][temp_RSI_name][temp_len-5-num_samples:temp_len-5].isnull().any():")
                        # print(str(elaborated_data_series[count][temp_RSI_name][temp_len-5-num_samples:temp_len-5].isnull().any()))
                        # print("RSI cycles analysis, temp_indicator_period: " + str(temp_indicator_period))

                        num_samples = int(temp_indicator_period * multiple_factor * 4)
                        start_index = end_rebuilt_signal_index - num_samples


                        # print('smooted_end_date: ' + str(smooted_end_date))

                        # min_period = int( temp_period - 2*temp_period/3)
                        min_period = int( temp_indicator_period - 2*temp_indicator_period/3)
                        if(min_period < 6):
                            min_period = 6

                        max_period = int(temp_indicator_period * multiple_factor)

                        RSI_column_name = temp_RSI_name
                        RSI_current_date = current_date

                        if(RSI_cycles_analysis_type == 'SG_smooted_RSI'):
                            RSI_column_name = temp_smooted_RSI_name
                            RSI_current_date = self.data.index.date[end_rebuilt_signal_index - int(temp_indicator_period*2/3)]
                            RSI_current_date = str(RSI_current_date)


                        # Cycles analysis for RSI
                        _, _, temp_elaborated_data, _, _ =  self.analyze_and_plot(
                                    data =  elaborated_data_series[count],
                                    data_column_name = RSI_column_name, #temp_RSI_name,
                                    transform_precision = 0.01,
                                    num_samples = num_samples,
                                    start_date = None,
                                    current_date = RSI_current_date, #str(smooted_end_date), # current_date,
                                    final_kept_n_dominant_circles = 5,
                                    dominant_cicles_sorting_type = 'global_score',
                                    limit_n_harmonics = None,
                                    min_period = min_period,
                                    max_period = max_period,
                                    detrend_type = 'hp_filter',
                                    lowess_k = lowess_k,
                                    bartel_peaks_filtering = True,
                                    bartel_scoring_threshold = 0,
                                    hp_filter_lambda = 1000 * 10 ** ( temp_indicator_period * multiple_factor / 20 ),
                                    jp_filter_p = 4,
                                    jp_filter_h = 8,
                                    centered_averages = True,
                                    time_zone = 'America/New_York',
                                    other_correlations = True,
                                    include_calibrated_MACD = False,
                                    include_calibrated_RSI = False,
                                    show_charts = False,
                                    print_report = False,
                                    indicators_signal_calcualtion = False,
                                    debug = False,
                                    time_tracking = time_tracking)

                        if(temp_elaborated_data is not None):

                            rsi_min = min(elaborated_data_series[count][RSI_column_name][start_index:end_rebuilt_signal_index+1])
                            rsi_max = max(elaborated_data_series[count][RSI_column_name][start_index:end_rebuilt_signal_index+1])

                            len_rsi_CDC = len(temp_elaborated_data['composite_dominant_circles_signal'])
                            if( len_rsi_CDC > max_series_len):
                                len_rsi_CDC = max_series_len

                            RSI_scaler = MinMaxScaler(feature_range=(rsi_min, rsi_max))

                            # print('\nlen(cdc_rsi): ' + str(len(cdc_rsi)))
                            # print('start_index: ' + str(start_index))
                            # print('len_rsi_CDC: ' + str(len_rsi_CDC))

                            cdc_rsi['CDC_' + temp_smooted_RSI_name] = pd.Series([float('nan')] * len(cdc_rsi))
#                             cdc_rsi['CDC_' + temp_smooted_RSI_name].iloc[start_index:len_rsi_CDC] = RSI_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:len_rsi_CDC].values.reshape(-1, 1)).flatten()
                            cdc_rsi.loc[start_index:len_rsi_CDC, 'CDC_' + temp_smooted_RSI_name] = RSI_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:len_rsi_CDC].values.reshape(-1, 1)).flatten()


                        else:
                            if(self.print_activity_remarks):
                                print('\t\tError in cycles analysis for cdc_rsi')
                                print('\t\ttemp_indicator_period: ' + str(temp_indicator_period))
                                print(f'\t\tmin_period: {min_period}, max_period: {max_period}')

                            # elaborated_data_series[count][temp_smooted_RSI_name],

                count += 1



        # -----------------------------------------------------------------------
        #                   Cycles analysis for savgol MACD
        # -----------------------------------------------------------------------

        if(CDC_MACD_analysis == True):

            if(self.print_activity_remarks):
                print('\nCycles analysis for savgol MACD')

            cdc_SG_MACD_signals = pd.DataFrame(index=elaborated_data_series[max_length_series_index].index)
            cdc_SG_MACD_hists = pd.DataFrame(index=elaborated_data_series[max_length_series_index].index)
            max_series_len = len(elaborated_data_series[max_length_series_index])
            count = 0

            for signal_serie in signals_results_series:

                for indicator_parameters in signal_serie['indicators_parameters']:

                    temp_savgol_MACD_signal_name = indicator_parameters['MACD_parameters']['savgol_MACD_signal_name']
                    temp_savgol_MACD_hist_name= indicator_parameters['MACD_parameters']['savgol_MACD_hist_name']
                    temp_indicator_period = indicator_parameters['MACD_parameters']['indicators_period']


                    if(temp_indicator_period >= 6):

                        if(self.print_activity_remarks):
                            print('\tPeriod: ' + str(temp_indicator_period))

                        num_samples = int(temp_indicator_period * multiple_factor * 4)
                        start_index = end_rebuilt_signal_index - num_samples
                        # last parte of savsgol filtered MACD is not trustable
                        smooted_end_date = str( self.data.index.date[end_rebuilt_signal_index - int(temp_indicator_period*2/3)] )

                        # min_period = int( temp_period - 2*temp_period/3)
                        min_period = int( temp_indicator_period - 2*temp_indicator_period/3)
                        if(min_period < 6):
                            min_period = 6

                        # Cycles analysis for savgol_MACD_signals
                        _, _, temp_elaborated_data, _, _ =  self.analyze_and_plot(
                                    data =  elaborated_data_series[count],
                                    data_column_name = temp_savgol_MACD_signal_name, #temp_RSI_name,
                                    transform_precision = 0.01,
                                    num_samples = num_samples,
                                    start_date = None,
                                    current_date = smooted_end_date, # current_date,
                                    final_kept_n_dominant_circles = 5,
                                    dominant_cicles_sorting_type = 'global_score',
                                    limit_n_harmonics = None,
                                    min_period = min_period,
                                    max_period = int(temp_indicator_period * multiple_factor),
                                    detrend_type = 'hp_filter',
                                    lowess_k = lowess_k,
                                    bartel_peaks_filtering = True,
                                    bartel_scoring_threshold = 0,
                                    hp_filter_lambda = 1000 * 10 ** ( temp_indicator_period * multiple_factor / 20 ),
                                    jp_filter_p = 4,
                                    jp_filter_h = 8,
                                    centered_averages = True,
                                    time_zone = 'America/New_York',
                                    other_correlations = True,
                                    include_calibrated_MACD = False,
                                    include_calibrated_RSI = False,
                                    show_charts = False,
                                    print_report = False,
                                    indicators_signal_calcualtion = False,
                                    debug = False,
                                    time_tracking = time_tracking)

                        if(temp_elaborated_data is not None):

                            MACD_signal_min = min(elaborated_data_series[count][temp_savgol_MACD_signal_name][start_index:end_rebuilt_signal_index+1])
                            MACD_signal_max = max(elaborated_data_series[count][temp_savgol_MACD_signal_name][start_index:end_rebuilt_signal_index+1])

                            len_MACD_CDC = len(temp_elaborated_data['composite_dominant_circles_signal'])
                            if( len_MACD_CDC > max_series_len):
                                len_MACD_CDC = max_series_len

                            MACD_signal_scaler = MinMaxScaler(feature_range=(MACD_signal_min, MACD_signal_max))

                            cdc_SG_MACD_signals['CDC_' + temp_savgol_MACD_signal_name] = pd.Series([float('nan')] * len(cdc_SG_MACD_signals))
                            cdc_SG_MACD_signals['CDC_' + temp_savgol_MACD_signal_name].iloc[start_index:len_MACD_CDC] = MACD_signal_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:len_MACD_CDC].values.reshape(-1, 1)).flatten()

                        else:
                            if(self.print_activity_remarks):
                                print('\t\tError in cycles analysis for cdc_SG_MACD_signals')
                                print('\t\ttemp_indicator_period: ' + str(temp_indicator_period))


                        # Cycles analysis for savgol_MACD_hists
                        _, _, temp_elaborated_data, _, _ =  self.analyze_and_plot(
                                    data =  elaborated_data_series[count],
                                    data_column_name = temp_savgol_MACD_hist_name,
                                    transform_precision = 0.01,
                                    num_samples = num_samples,
                                    start_date = None,
                                    current_date = smooted_end_date, # current_date,
                                    final_kept_n_dominant_circles = 5,
                                    dominant_cicles_sorting_type = 'global_score',
                                    limit_n_harmonics = None,
                                    min_period = min_period,
                                    max_period = int(temp_indicator_period * multiple_factor),
                                    detrend_type = 'hp_filter',
                                    lowess_k = lowess_k,
                                    bartel_peaks_filtering = True,
                                    bartel_scoring_threshold = 0,
                                    hp_filter_lambda = 1000 * 10 ** ( temp_indicator_period * multiple_factor / 20 ),
                                    jp_filter_p = 4,
                                    jp_filter_h = 8,
                                    centered_averages = True,
                                    time_zone = 'America/New_York',
                                    other_correlations = True,
                                    include_calibrated_MACD = False,
                                    include_calibrated_RSI = False,
                                    show_charts = False,
                                    print_report = False,
                                    indicators_signal_calcualtion = False,
                                    debug = False,
                                    time_tracking = time_tracking)

                        if(temp_elaborated_data is not None):

                            MACD_hist_min = min(elaborated_data_series[count][temp_savgol_MACD_hist_name][start_index:end_rebuilt_signal_index+1])
                            MACD_hist_max = max(elaborated_data_series[count][temp_savgol_MACD_hist_name][start_index:end_rebuilt_signal_index+1])

                            len_MACD_CDC = len(temp_elaborated_data['composite_dominant_circles_signal'])
                            if( len_MACD_CDC > max_series_len):
                                len_MACD_CDC = max_series_len

                            MACD_hist_scaler = MinMaxScaler(feature_range=(MACD_hist_min, MACD_hist_max))

                            cdc_SG_MACD_hists['CDC_' + temp_savgol_MACD_hist_name] = pd.Series([float('nan')] * len(cdc_SG_MACD_hists))
                            cdc_SG_MACD_hists['CDC_' + temp_savgol_MACD_hist_name].iloc[start_index:len_MACD_CDC] = MACD_hist_scaler.fit_transform(temp_elaborated_data['composite_dominant_circles_signal'][start_index:len_MACD_CDC].values.reshape(-1, 1)).flatten()

                        else:
                            if(self.print_activity_remarks):
                                print('\t\tError in cycles analysis for cdc_SG_MACD_hists')
                                print('\t\ttemp_indicator_period: ' + str(temp_indicator_period))


                count += 1


        # -------------------------------------------------
        #               SCALED SIGNALS
        # -------------------------------------------------
        
        if(self.print_activity_remarks):
            print("\nSignals scaling")
            self.track_time('\n11. Genetics end composite signal creation')

        scaled_signals = pd.DataFrame()

        CDC_min = elaborated_data_series[index_detrended_data]['detrended'][max_start_index:].min(skipna=True)
        CDC_max = elaborated_data_series[index_detrended_data]['detrended'][max_start_index:].max(skipna=True)


        CDC_scaler = MinMaxScaler(feature_range=(CDC_min , CDC_max ))
#
#         print("len composite_signal['composite_signal']: " + str(len(composite_signal['composite_signal'])))
#         print("len alignmentsKPI: " + str(len(alignmentsKPI)))
#         print(f'len scaled_signals {len(scaled_signals)}') 
#         print(f'len elaborated_data_series[index_detrended_data] {len(elaborated_data_series[index_detrended_data])}') 
#         print(f'index_detrended_data {index_detrended_data}')
        
        
#         # Itera su tutti gli indici presenti in elaborated_data_series
#         for i, series in enumerate(elaborated_data_series):
#             if isinstance(series, pd.DataFrame):
#                 if 'detrended' in series:
#                     detrended_len = len(series['detrended'].values)
#                     print(f"Lunghezza di 'detrended' per index {i}: {detrended_len}")
#                 else:
#                     print(f"'detrended' non trovato per index {i}")
#             else:
#                 print(f"Elemento {i} non è un DataFrame")

        scaled_signals['scaled_composite_signal'] = scaled_composite_signal  = CDC_scaler.fit_transform( composite_signal['composite_signal'].values.reshape(-1, 1) ).flatten()
        scaled_signals['scaled_goertzel_composite_signal'] = scaled_goertzel_composite_signal = CDC_scaler.fit_transform( composite_signal['goertzel_composite_signal'].values.reshape(-1, 1) ).flatten()
        
        # Calcola la differenza di lunghezza tra scaled_signals e detrended
        detrended_values = elaborated_data_series[index_detrended_data]['detrended'].values
        scaled_signals_len = len(scaled_signals['scaled_composite_signal'])  # Assumiamo che sia questa la lunghezza target

        # Se detrended_values è più corto, aggiungi NaN all'inizio
        if len(detrended_values) < scaled_signals_len:
            num_nans_to_add = scaled_signals_len - len(detrended_values)
            detrended_values = np.concatenate([np.full(num_nans_to_add, np.nan), detrended_values])

        # Assegna i valori corretti
        scaled_signals['scaled_detrended'] = scaled_detrended  = detrended_values


#         scaled_signals['scaled_detrended'] = scaled_detrended = elaborated_data_series[index_detrended_data]['detrended'].values
        
        if(enable_cycles_alignment_analysis == True):

            scaled_signals['scaled_alignmentsKPI'] = scaled_alignmentsKPI = scaler.fit_transform( alignmentsKPI.values.reshape(-1, 1)).flatten()
            scaled_signals['scaled_weigthed_alignmentsKPI'] = scaled_weigthed_alignmentsKPI = scaler.fit_transform( weigthed_alignmentsKPI.values.reshape(-1, 1)).flatten()
            
        else:
            scaled_signals['scaled_alignmentsKPI'] = scaled_alignmentsKPI = np.zeros(len(scaled_signals['scaled_composite_signal']))
            scaled_signals['scaled_weigthed_alignmentsKPI'] = scaled_weigthed_alignmentsKPI = np.zeros(len(scaled_signals['scaled_composite_signal']))

        if(self.print_activity_remarks):
            self.track_time('12. Genetics end composite signal creation')


        # -------------------------------------------------
        #               CHARTS PLOT
        # -------------------------------------------------

        if(self.print_activity_remarks):
            print('Before CHARTS PLOT')
        
        if(show_charts == True):


            # Original data, detrended, dominant circles signal, averages delta
            fig = make_subplots(rows=9,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=("Original Data - " + self.ticker + " " + data_column_name + " Price",
                                                                                                          "Composite Domaninant Cycles Signal",
                                                                                                          "Cycles Alignment Indicators",
                                                                                                          "Dominant Cycles",
                                                                                                          "BB Deltas",
                                                                                                          "BB Daltas Der^1",
                                                                                                          "BB Daltas Der^2",
                                                                                                          'RSI and Savgol Smooted RSI',
                                                                                                          'Savsgol MACD'))


            fig.add_trace(go.Scatter(x=self.data.index,
                                      y=self.data[data_column_name],
                                      mode="lines",
                                      name="Original data"),
                          row=1,
                          col=1)

            missing_values = composite_signal['composite_signal'].isnull().any()

            if missing_values:
                print("Ci sono valori mancanti nella serie.")


            fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                    y= scaled_composite_signal,
                                    mode="lines",
                                    name="Composite Domaninant Cycles Signal GA Refactored"),
                          row=2, col=1)


            fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                    y = scaled_goertzel_composite_signal,
                                    mode="lines",
                                    name="Composite Domaninant Cycles Signal Goertzel Refactored"),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                    y= scaled_detrended,
                                    mode="lines",
                                    name="Detrended Signal (max lambda, minimal detrended)"),
                          row=2, col=1)

            if(MultiAn_fitness_type_svg_smoothed == True):
                fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                        y=scaler.fit_transform( self.MultiAn_reference_detrended_data.values.reshape(-1, 1)).flatten() ,
                                        mode="lines",
                                        name="Smoothed Detrended Signal (max lambda, minimal detrended)"),
                              row=2, col=1)


            # Add a vertical line spanning the height of the chart
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=scaled_goertzel_composite_signal.min(),
                y1=scaled_goertzel_composite_signal.max(),
                line=dict(color='purple', width=1),
                row=2, col=1
            )

            # scaled_alignmentsKPI = scaler.fit_transform( alignmentsKPI.values.reshape(-1, 1)).flatten()
            # scaled_weigthed_alignmentsKPI = scaler.fit_transform( weigthed_alignmentsKPI.values.reshape(-1, 1)).flatten()



            fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                    y = scaled_alignmentsKPI, #  alignmentsKPI,
                                    mode = "lines",
                                    name = "Cycles Alignment Indicator"),
                          row=3, col=1)


            fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                     y = scaled_weigthed_alignmentsKPI, # weigthed_alignmentsKPI,
                                     mode="lines",
                                     name="Weigthed Cycles Alignment Indicator",
                                     yaxis="y2"),
                          row=3, col=1)


            # Add a vertical line spanning the height of the chart
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=scaled_weigthed_alignmentsKPI.min(),
                y1=scaled_weigthed_alignmentsKPI.max(),
                line=dict(color='purple', width=1),
                row=3, col=1
            )


            y_min = 0
            y_max = 0

            for index_main in range(len(signals_results_series)):

                signals_results_series_row = signals_results_series[index_main]

                for index_inner in range(len(signals_results_series_row['dominant_peaks_signals'])):

                    row_inner = signals_results_series_row['dominant_peaks_signals'][index_inner]

                    period_name = str(row_inner['peak_periods'])

                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                            y=composite_signal['composite_signal_refact_dominant_circle_signal_period_' + period_name] ,
                                            mode="lines",
                                            name="Dominant Circle " + period_name),
                                  row=4, col=1)

                    if(composite_signal['composite_signal_refact_dominant_circle_signal_period_' + period_name].max(skipna=True) > y_max):
                        y_max = composite_signal['composite_signal_refact_dominant_circle_signal_period_' + period_name].max(skipna=True)

                    if(composite_signal['composite_signal_refact_dominant_circle_signal_period_' + period_name].min(skipna=True) < y_min):
                        y_min =  composite_signal['composite_signal_refact_dominant_circle_signal_period_' + period_name].min(skipna=True)


            # Add a vertical line spanning the height of the chart
            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=y_min,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=4, col=1
            )


            y_max = 0
            for temp_period in BB_periods:

                fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                        y= bb_delta['BB_delta_' + str(temp_period)],
                                        mode="lines",
                                        name='BB_delta_' + str(temp_period)),
                              row=5, col=1)

                if(CDC_bb_analysis == True):
                    
                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                            y= bb_delta['BB_delta_CDC_' + str(temp_period)] ,
                                            mode="lines",
                                            name='BB_delta_CDC_' + str(temp_period)),
                                  row=5, col=1)

                temp_max = bb_delta['BB_delta_' + str(temp_period)].max(skipna=True)

                if(temp_max > y_max):
                    y_max =  temp_max



            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=0,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=5, col=1
            )

            y_max = 0
            y_min = 0
            for temp_period in BB_periods:

                fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                        y= bb_delta['BB_delta_der^1_' + str(temp_period)],
                                        mode="lines",
                                        name='BB_delta_der^1_' + str(temp_period)),
                              row=6, col=1)

                fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                        y= bb_delta['BB_delta_CDC_der^1_' + str(temp_period)],
                                        mode="lines",
                                        name='BB_delta_CDC_der^1_' + str(temp_period)),
                              row=6, col=1)

                temp_max = bb_delta['BB_delta_der^1_' + str(temp_period)].max(skipna=True)

                if(temp_max > y_max):
                    y_max =  temp_max

                temp_min = bb_delta['BB_delta_der^1_' + str(temp_period)].min(skipna=True)

                if(temp_min < y_min):
                    y_min =  temp_min


            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=y_min,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=6, col=1
            )



            for temp_period in BB_periods:

                fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                        y= bb_delta['BB_delta_der^2_' + str(temp_period)],
                                        mode="lines",
                                        name='BB_delta_der^2_' + str(temp_period)),
                              row=7, col=1)


            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=y_min,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=7, col=1
            )


            # -------------------------------------------------------
            #       RSI, SAVSGOL SMOOTED RSI AND CDC RSI PLOTS
            # -------------------------------------------------------

            count = 0
            for signal_serie in signals_results_series:

                for indicator_parameters in signal_serie['indicators_parameters']:

                    temp_RSI_name = indicator_parameters['RSI_parameters']['RSI_name']
                    temp_smooted_RSI_name = indicator_parameters['RSI_parameters']['smoothed_RSI_name']

                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                          y= elaborated_data_series[count][temp_RSI_name],
                                          mode="lines",
                                          name=temp_RSI_name),
                                row=8, col=1)

                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                          y= elaborated_data_series[count][temp_smooted_RSI_name],
                                          mode="lines",
                                          name=temp_smooted_RSI_name),
                                row=8, col=1)

                count += 1
                
            if(CDC_RSI_analysis == True):
                for column in cdc_rsi.columns:

                      fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                              y= cdc_rsi[column],
                                              mode="lines",
                                              name=column),
                                    row=8, col=1)

            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=y_min,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=8, col=1
            )


            # -------------------------------------------------------
            #       SAVSGOL MACD AND CDC MACD PLOTS
            # -------------------------------------------------------


            count = 0
            for signal_serie in signals_results_series:

                for indicator_parameters in signal_serie['indicators_parameters']:

                    # temp_MACD_name = indicator_parameters['MACD_parameters']['macd_name']
                    # temp_MACD_signal_name = indicator_parameters['MACD_parameters']['macdSignal_name']
                    # temp_MACD_hist_name = indicator_parameters['MACD_parameters']['macdHist_name']

                    temp_MACD_signal_name = indicator_parameters['MACD_parameters']['savgol_MACD_signal_name']
                    temp_MACD_hist_name = indicator_parameters['MACD_parameters']['savgol_MACD_hist_name']

                    # fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                    #                         y= elaborated_data_series[count][temp_MACD_name],
                    #                         mode="lines",
                    #                         name=temp_MACD_name),
                    #               row=9, col=1)
                    
#                     print(f"\ncount = {count}")
#                     print(f"temp_MACD_signal_name = {temp_MACD_signal_name}")
#                     print(f"available columns = {elaborated_data_series[count].columns}")
#                     print(f"indicator_parameters: {indicator_parameters}")

                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                          y= elaborated_data_series[count][temp_MACD_signal_name],
                                          mode="lines",
                                          name=temp_MACD_signal_name),
                                row=9, col=1)

                    fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                          y= elaborated_data_series[count][temp_MACD_hist_name],
                                          mode="lines",
                                          name=temp_MACD_hist_name),
                                row=9, col=1)

                count += 1
                    
            if(CDC_MACD_analysis == True):
                
                for column in cdc_SG_MACD_signals.columns:

                      fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                              y= cdc_SG_MACD_signals[column],
                                              mode="lines",
                                              name=column),
                                    row=9, col=1)

                for column in cdc_SG_MACD_hists.columns:

                      fig.add_trace(go.Scatter(x=elaborated_data_series[max_length_series_index].index,
                                              y= cdc_SG_MACD_hists[column],
                                              mode="lines",
                                              name=column),
                                    row=9, col=1)


            fig.add_shape(
                type='line',
                x0=index_of_max_time_for_cd,
                x1=index_of_max_time_for_cd,
                y0=y_min,
                y1=y_max,
                line=dict(color='purple', width=1),
                row=9, col=1
            )


            # fig.update_yaxes(title_text="Cycles Alignment Indicators", secondary_y=True, row=3, col=1)
            fig.update_layout(title="Goertzel Dominant Circles Analysis", height=2200)

            fig.update_xaxes(type="category")

            # Visualizza il secondo grafico con i subplot
            fig.show()


        return elaborated_data_series, signals_results_series, composite_signal, configurations_series, bb_delta, cdc_rsi, index_of_max_time_for_cd, scaled_signals, best_fitness_value



    def get_goertzel_amplitudes(self):

        return self.goertzel_amplitudes


    def cicles_composite_signals(self, max_length_series, amplitudes, MultiAn_dominant_cycles_df, df_indexes_list, composite_signal_column_name):
        
#         print('\nIn cicles_composite_signals')
#         print('amplitudes')
#         display(amplitudes)
#         print('MultiAn_dominant_cycles_df')
#         display(MultiAn_dominant_cycles_df)

        composite_signal = pd.DataFrame(index=df_indexes_list)
        composite_signal[composite_signal_column_name] = 0.0

        max_start_rebuilt_signal_index = 0

        for index, row in MultiAn_dominant_cycles_df.iterrows():

            temp_period = row['peak_periods'] #  1/row['peak_frequencies']
            new_column_name = composite_signal_column_name + '_refact_dominant_circle_signal_period_' + str(temp_period)
            composite_signal[new_column_name] = 0.0
            # composite_signal['temp_goertzel_refactored_cycle'] = 0

            remanant_length = np.int64(max_length_series - row['start_rebuilt_signal_index'])
            time = np.linspace(0, remanant_length, remanant_length, endpoint=False)

            # Composite Cycles from GA amplitudes refactoring
            # cicle_signal = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
            composite_signal.iloc[int(row['start_rebuilt_signal_index']):, composite_signal.columns.get_loc(new_column_name)] = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases']) #[new_column_name].iloc[row['start_rebuilt_signal_index']:] = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
            composite_signal[composite_signal_column_name]  += composite_signal[new_column_name]

            # # Composite Cycles from Goertzel transform amplitudes refactoring
            # composite_signal['temp_goertzel_refactored_cycle'].iloc[row['start_rebuilt_signal_index']:] = goertzel_amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])
            # composite_signal['goertzel_composite_signal']  += composite_signal['temp_goertzel_refactored_cycle']
           

            if(max_start_rebuilt_signal_index < int(row['start_rebuilt_signal_index'])):
                max_start_rebuilt_signal_index = int(row['start_rebuilt_signal_index'])
                
#             print(f"- period: {temp_period}, start_rebuilt_signal_index: {row['start_rebuilt_signal_index']}")

        print(f"composite_signal {composite_signal}.iloc[int(row['start_rebuilt_signal_index']):]")
        return composite_signal



    def indict_MACD_SGMACD(self, data, signals_results, data_column_name, dominant_period, macd_slow_ratio = 26, macd_signal_ratio = 9):

        indicators_period = int(dominant_period / 2)
        macd_fast = indicators_period
        macd_slow = int(macd_slow_ratio/12 * macd_fast)
        macd_signal = int(macd_signal_ratio/12 * macd_fast)

        data['macd_' + str(indicators_period)], data['macdSignal_' + str(indicators_period)], data['macdHist_' + str(indicators_period)] = talib.MACD(data[data_column_name], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)

        data['macdSignal_derivate_' + str(indicators_period)] = data['macdHist_' + str(indicators_period)].diff()
        data['macdSignal_derivate_' + str(indicators_period)] = data['macdSignal_derivate_' + str(indicators_period)] .fillna(0)

        # savgol MACD
        savgol_filter_long_period = indicators_period*4
        savgol_filter_short_period = indicators_period*2

        data['savgol_filter_long_' + str(indicators_period)] = savgol_filter(data[data_column_name], int(savgol_filter_long_period), 2)
        data['savgol_filter_short_' + str(indicators_period)] = savgol_filter(data[data_column_name], int(savgol_filter_short_period), 2)
        data['savgol_MACD_' + str(indicators_period)] = data['savgol_filter_short_' + str(indicators_period)] - data['savgol_filter_long_' + str(indicators_period)]
        data['savgol_MACD_signal_' + str(indicators_period)] = savgol_filter(data['savgol_MACD_' + str(indicators_period)] , int(indicators_period*2), 2)
        data['savgol_MACD_hist_' + str(indicators_period)] = data['savgol_MACD_' + str(indicators_period)] - data['savgol_MACD_signal_' + str(indicators_period)]

        #signals_results['dominant_scaled_amplitude_analysis']['MACD_' + str(indicators_period)]
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


    def indict_RSI_SG_smooth_RSI(self, data, signals_results, data_column_name, end_rebuilt_signal_index, dominant_period = 10):

        data_len = len(data)
        indicators_period = int(dominant_period / 2)
        data['RSI_' + str(indicators_period)] = talib.RSI(data[data_column_name], indicators_period)
        data['RSI_' + str(indicators_period)] = data['RSI_' + str(indicators_period)] .fillna(0)

        polyorder = 2
        if(indicators_period < 3):
            polyorder = 1

        data['smoothed_RSI_' + str(indicators_period)] = pd.Series([np.nan] * data_len)
        data.iloc[0:end_rebuilt_signal_index+1, data.columns.get_loc('smoothed_RSI_' + str(indicators_period))] = savgol_filter(data['RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1], indicators_period, polyorder)  #   ['smoothed_RSI_' + str(indicators_period)].iloc[0:end_rebuilt_signal_index+1] = savgol_filter(data['RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1], indicators_period, polyorder)
        # data['smoothed_RSI_' + str(indicators_period)] = data['smoothed_RSI_' + str(indicators_period)]

        data['smoothed_RSI_derivate_' + str(indicators_period)] = pd.Series([np.nan] * data_len)
        data['smoothed_RSI_derivate_' + str(indicators_period)] = data['smoothed_RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1].diff()
        # data['smoothed_RSI_derivate_' + str(indicators_period)] = data['smoothed_RSI_derivate_' + str(indicators_period)]

        # signals_results['dominant_scaled_amplitude_analysis']['RSI_' + str(indicators_period)]
        indicator_parameters = {
                                'dominant_period': dominant_period,
                                'indicator_period': indicators_period,
                                'RSI_name': 'RSI_' + str(indicators_period),
                                'smoothed_RSI_name': 'smoothed_RSI_' + str(indicators_period),
                                'smoothed_RSI_derivate_name': 'smoothed_RSI_derivate_' + str(indicators_period)
                               }

        return data, indicator_parameters
    
    def custom_crossover(self, ind1, ind2):
        
        if len(ind1) > 1 and len(ind2) > 1:
            # Esegui il crossover a due punti
            return tools.cxTwoPoint(ind1, ind2)
        else:
            # Gestisci caso di individui con un solo gene
            # Ad esempio, non fare nulla o applica una mutazione
            return tools.cxUniform(ind1, ind2, 0.5)



    def indict_centered_average_deltas(self, data, signals_results, data_column_name, dominant_period = 10):

        long_average = data[data_column_name].rolling(window=round(dominant_period), center = True).mean()
        long_average.fillna(0, inplace=True)
        short_average = data[data_column_name].rolling(window=round(dominant_period/2), center = True).mean()
        short_average.fillna(0, inplace=True)

        averages_delta = long_average - short_average

        delta_column_name = 'centered_averages_delta_' + str(round(dominant_period))

        data[delta_column_name] = averages_delta


#         signals_results['dominant_scaled_amplitude_analysis'][delta_column_name]
        indicator_parameters =  {
                                    'dominant_period': dominant_period,
                                    'long_average_period': round(dominant_period),
                                    'short_average_period': round(dominant_period/2),
                                    'centered_averages_delta_name': 'centered_averages_delta_' + str(round(dominant_period))
                                }

        return data, indicator_parameters


    def rebuilt_signal_zeros(self, signal, start_rebuilt_signal_index, data, debug = False):

        total_length = len(data)

        rebuilt_sig_left_zeros = np.zeros(start_rebuilt_signal_index ) #+ 1
        signal = np.concatenate((rebuilt_sig_left_zeros, signal), axis = 0)
        
        if(debug):
            np.set_printoptions(threshold=np.inf)
            print('signal before')
            display(signal)

        rebuilt_sig_right_zeros = 0

        # Add zeros on the right side if signal end before the last original_data element
        if(total_length  > len(signal)):

            rebuilt_sig_right_zeros = np.zeros(total_length - len(signal))
            signal = np.concatenate((signal, rebuilt_sig_right_zeros), axis = 0)
            # this ensures to limit the added rebuilt signal to the max current time (last sample index of original data)
            signal = signal[0:(total_length)]

        # if the projection exceed the original data index, calculated the extension length
        projection_periods_extetions = 0
        if(len(signal) > total_length):
            projection_periods_extetions = len(signal) - len(data)
            
        if(debug):
            print('signal after')
            display(signal)


        return signal, projection_periods_extetions


    def goertzel_DFT(self, testdata, testcycle_length, debug = False):

#         print("\ntest data type: " +str(type(testdata)))

        if isinstance(testdata, pd.Series):
            testdata = testdata.to_numpy()


        data_length = len(testdata)

        real = 0
        imag = 0
        r1, i1 = 0, 0
        SN, CN = 0, 0
        test_freq = 1 / testcycle_length
        coeff = 2.0 * math.cos(2.0 * math.pi * test_freq)
        Q0, Q1, Q2, omega, sine, cosine, temp = 0, 0, 0, 0, 0, 0, 0


        for i in range(data_length):
            if((debug != False) & (data_length < 95)):
                print("\ni: " + str(i))
                print("\tcoeff: " + str(coeff))
                print("\tdata_length: " + str(data_length))
                print("\ttestdata[i]: " + str(testdata[i]))
                print("\tQ0: " + str(Q0))
                print("\tQ1: " + str(Q1))
                print("\tQ2: " + str(Q2))
                
            Q0 = coeff * Q1 - Q2 + testdata[i] #.iloc[i]
            Q2 = Q1
            Q1 = Q0

        if((debug != False) & (data_length < 95)):
            print("\ni: " + str(i))
            print("\tdata_length: " + str(data_length))
            print("\ttestdata[i]: " + str(testdata[i]))
            print("\tQ0h: " + str(Q0))
            print("\tQ1: " + str(Q1))
            print("\tQ2: " + str(Q2))

        r1 = Q1 - Q2 * math.cos(2.0 * math.pi * test_freq)
        i1 = Q2 * math.sin(2.0 * math.pi * test_freq)
        CN = math.cos((data_length - 1) * 2.0 * math.pi * test_freq)
        SN = math.sin((data_length - 1) * 2.0 * math.pi * test_freq)
        real = r1 * CN + i1 * SN
        imag = i1 * CN - r1 * SN
        amp2 = (2.0 * math.sqrt(real**2 + imag**2)) #/ data_length
        phase2 = math.pi / 2 + math.atan2(imag, real)
        temp = phase2 / (2.0 * math.pi)
        minoffset = testcycle_length * ((math.pi + (math.pi / 2)) / (2 * math.pi) - temp)

        residual_t = data_length % testcycle_length
        argument = 2 * math.pi * (1 / testcycle_length) * residual_t + phase2

        # print("\nNew loop:")

        if(argument <= math.pi/2):
            # print("\tPOINT 1")
            maxoffset = round( ( (math.pi / 2) - argument ) * testcycle_length / ( 2 * math.pi) )
        elif(math.pi/2 < argument <= (math.pi * 5 / 2)):
            # print("\tPOINT 2")
            maxoffset = round( ( (math.pi * 5 / 2) - argument ) * testcycle_length / ( 2 * math.pi) )
        elif((math.pi * 5 / 2) < argument):
            # print("\tPOINT 3")
            maxoffset = round( ( (math.pi * 9 / 2) - argument ) * testcycle_length / ( 2 * math.pi) )
        else:


            print("\tdata_length: " + str(data_length))
            print("\ttestcycle_length: " + str(testcycle_length))
            print("\tresidual_t: " + str(residual_t))
            print("\tphase2: " + str(phase2))
            print("\targument: " + str(argument))
            print("\treal: " + str(real))
            print("\timag: " + str(imag))
            print("\tCN: " + str(CN))
            print("\tSN: " + str(SN))
            print("\tr1: " + str(r1))
            print("\ti1: " + str(i1))
            print("\ttest_freq: " + str(test_freq))
            print("\tQ0: " + str(Q0))
            print("\tQ1: " + str(Q1))
            print("\tQ2: " + str(Q2))

        if(math.isnan(residual_t) or residual_t is None):

            print("\t\tTestdata[i] == Nan")
            print("\t\t count Nan: " + str(np.isnan(testdata).sum()))
            print(testdata)


        if(maxoffset < 0):
            print("argument: " + str(argument))
            print("maxoffset: " + str(maxoffset))

        if(argument < (math.pi * 3 / 2)):
            minoffset2 = round( ( (math.pi * 3 / 2) - argument ) * testcycle_length / ( 2 * math.pi) )

        else:
            minoffset2 = round( ( (math.pi * 7 / 2) - argument ) * testcycle_length / ( 2 * math.pi) )

        return amp2, phase2, minoffset, minoffset2, maxoffset



    def get_row_score(self, row):

        scores = row.sum()

        return scores



    def get_gloabl_score(self, data, ascending_columns, descending_columns):

        df = pd.DataFrame(data)
        data_ascending = pd.DataFrame()
        data_descending = pd.DataFrame()
        global_score = pd.DataFrame()

        # Seleziona solo le colonne desiderate da data
        # data_ascending = data[ascending_columns]
        # data_descending = data[descending_columns]

        data_ascending = df[ascending_columns].rank(ascending=True, axis=0)
        data_descending = df[descending_columns].rank(ascending=False, axis=0)

        global_score = pd.concat([data_ascending, data_descending], axis=1)
        global_score['global_score'] = global_score.apply(self.get_row_score, axis=1)

        return global_score['global_score']



    def trade_predicted_dominant_cicles_peaks(self,
                                              current_date,
                                              data,
                                              num_samples= 400,
                                              final_kept_n_dominant_circles=4,
                                              min_period = 30,
                                              max_period = 80,
                                              hp_filter_lambda = 170000,
                                              time_tracking = False):


        _, _, elaborated_data, _, _ = self.analyze_and_plot(
                                                data_column_name = 'Close',
                                                num_samples= num_samples,
                                                start_date= None, # '2007-09-01', #2023-09-12' '2023-09-28'
                                                current_date = current_date, #'2023-03-21', #'2009-02-25', # '2022-12-01', # '2022-10-03', 08-18,
                                                final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                dominant_cicles_sorting_type = 'global_score',
                                                limit_n_harmonics = None,
                                                min_period = min_period,
                                                max_period = max_period,
                                                detrend_type = 'hp_filter',
                                                bartel_scoring_threshold = 0,
                                                hp_filter_lambda = hp_filter_lambda, #170000
                                                jp_filter_p = 3,
                                                jp_filter_h = 100,
                                                bartel_peaks_filtering = True,
                                                centered_averages = True,
                                                time_zone = 'America/New_York',
                                                other_correlations = True,
                                                include_calibrated_MACD = False,
                                                include_calibrated_RSI = False,
                                                show_charts = False,
                                                print_report = False,
                                                indicators_signal_calcualtion = False,
                                                time_tracking = time_tracking
                                               )

        if elaborated_data is None:
            value = 0

        else:

            maxes = argrelmax(elaborated_data['composite_dominant_circles_signal'].to_numpy(), order = 10)[0]
            mins = argrelmin(elaborated_data['composite_dominant_circles_signal'].to_numpy(), order = 10)[0]
            current_date = current_date.replace(tzinfo=None)
            current_date_index = elaborated_data.index.get_loc(current_date)

            maxes = maxes[maxes > current_date_index]
            if(len(maxes) > 0):
                first_max = maxes[0]
            else:
                return 0

            mins = mins[mins > current_date_index]
            if(len(mins) > 0):
                first_min = mins[0]
            else:
                return 0

            value = elaborated_data['Close'].iloc[first_max+1] - elaborated_data['Close'].iloc[first_min+1]

            if(pd.isna(value) or value == None):
                return 0

            else:
                return value





    def CDC_vs_detrended_correlation(self,
                                     current_date,
                                     data,
                                     num_samples,
                                     final_kept_n_dominant_circles,
                                     min_period,
                                     max_period,
                                     hp_filter_lambda,
                                     opt_algo_type = 'genetic_omny_frequencies', # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency', 'tpe', 'atpe'
                                     detrend_type = 'hp_filter', #'linear', #'hp_filter',
                                     lowess_k = 3,
                                     linear_filter_window_size_multiplier = 0.7,
                                     period_related_rebuild_range = True,
                                     period_related_rebuild_multiplier = 2.5,
                                     best_fit_start_back_period = None, #16*2, #  None
                                     ):
        
# NB: aggiungere parametro, il confronto farlo su un numero di periodi diverso dalla trasfromata, potenzialmente più piccolo, più vicino
#     ai valori attuali

#         current_date, index_of_max_time_for_cd, elaborated_data, signals_results, configuration = self.analyze_and_plot(
#                                                 data_column_name = 'Close',
#                                                 num_samples= num_samples,
#                                                 start_date= None,
#                                                 current_date = current_date,
#                                                 final_kept_n_dominant_circles=final_kept_n_dominant_circles,
#                                                 dominant_cicles_sorting_type = 'global_score',
#                                                 limit_n_harmonics = None,
#                                                 min_period = min_period,
#                                                 max_period = max_period,
#                                                 detrend_type = 'hp_filter',
#                                                 bartel_scoring_threshold = 0,
#                                                 hp_filter_lambda = hp_filter_lambda, #170000
#                                                 jp_filter_p = 3,
#                                                 jp_filter_h = 100,
#                                                 bartel_peaks_filtering = True,
#                                                 centered_averages = True,
#                                                 time_zone = 'America/New_York',
#                                                 other_correlations = True,
#                                                 include_calibrated_MACD = False,
#                                                 include_calibrated_RSI = False,
#                                                 show_charts = False,
#                                                 print_report = False,
#                                                 indicators_signal_calcualtion = False
#                                                )

        cycles_parameters = pd.DataFrame(columns = ['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda'])
        cycles_parameters.loc[len(cycles_parameters)] = [num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda]
    

#         print('Calling multiperiod_analysis(), parameters')
#         print(f'opt_algo_type = opt_algo_type: {opt_algo_type}')
#         print(f'linear_filter_window_size_multiplier: {linear_filter_window_size_multiplier}')
#         print(f'period_related_rebuild_range: {period_related_rebuild_range}')
#         print(f'period_related_rebuild_multiplier: {period_related_rebuild_multiplier}')
#         print(f'self.time_tracking: {self.time_tracking}')


        
        
        (elaborated_data_df, 
         signals_results_df, 
         composite_signal, 
         configurations, 
         bb_delta, cdc_rsi, 
         index_of_max_time_for_cd, 
         scaled_signals, 
         best_fitness_value) = self.multiperiod_analysis(
                                                        data_column_name = 'Close',
                                                        current_date = '2024-05-30',
                                                        periods_pars = cycles_parameters,   
                                                        time_zone = 'America/New_York',
                                                        pars_from_opt_file = False,
                                                        files_path_name = None,
                                                        population_n = 10,
                                                        CXPB = 0.7,
                                                        MUTPB = 0.3,
                                                        NGEN = 30,
                                                        MultiAn_fitness_type = "mse",
                                                        MultiAn_fitness_type_svg_smoothed = False,
                                                        MultiAn_fitness_type_svg_filter = 4,
                                                        reference_detrended_data = "less_detrended", # longest less_detrended
                                                        bb_delta_fixed_periods = [8, 16], #, 32],
                                                        bb_delta_sg_filter_window = None,
                                                        RSI_cycles_analysis_type = 'original_RSI', #'SG_smooted_RSI' # 'original_RSI',
                                                        enable_cycles_alignment_analysis = False,
                                                        opt_algo_type = opt_algo_type, #'genetic_omny_frequencies', # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency', 'tpe', 'atpe'
                                                        detrend_type = detrend_type, #'linear', #'linear', #'hp_filter',
                                                        linear_filter_window_size_multiplier = linear_filter_window_size_multiplier, #0.7,
                                                        period_related_rebuild_range = period_related_rebuild_range, #True,
                                                        period_related_rebuild_multiplier = period_related_rebuild_multiplier, #2.5,
                                                        best_fit_start_back_period = None, #16*2, #  None
                                                        CDC_bb_analysis = False,
                                                        CDC_RSI_analysis = False,
                                                        CDC_MACD_analysis = False,
                                                        show_charts = False,
                                                        enabled_multiprocessing = False, # double level multiprocessing is not allowed 
                                                        time_tracking = self.time_tracking
                                                        )

      # CONTROLLARE: perchè best_fit_start_back_period non è passato alla funzione di sopra. Qui si controlla solo su un numero di 
      # periodi pari a best_fit_start_back_period la fitness, ma l'elaborazione del segnale non sembra esser fatta ottimizzando 
      # questa porzione

    
#         if elaborated_data is None:
#             fitness = 0

#         else:

#             if(best_fit_start_back_period == None):
#                 composite_signal = elaborated_data['composite_dominant_circles_signal'][index_of_max_time_for_cd-num_samples:index_of_max_time_for_cd]
#                 detrended_signal = elaborated_data['detrended'][index_of_max_time_for_cd-num_samples:index_of_max_time_for_cd]
#                 print(f'MSE evalutin on num_samples = {num_samples}')
                
#             else:
#                 composite_signal = elaborated_data['composite_dominant_circles_signal'][index_of_max_time_for_cd-best_fit_start_back_period:index_of_max_time_for_cd]
#                 detrended_signal = elaborated_data['detrended'][index_of_max_time_for_cd-best_fit_start_back_period:index_of_max_time_for_cd]
#                 print(f'MSE evalutin on best_fit_start_back_period = {best_fit_start_back_period}')
                                

#             if(composite_signal.isna().any()):
#                 print('composite_signal contiene dei Nan')
#             if(detrended_signal.isna().any()):
#                 print('detrended_signal contiene dei Nan')

#             composite_signal = self.scaler.fit_transform(composite_signal.values.reshape(-1, 1)).flatten()
#             detrended_signal = self.scaler.fit_transform(detrended_signal.values.reshape(-1, 1)).flatten()

#             fitness = mean_squared_error(composite_signal,detrended_signal)


#         return fitness

        if(composite_signal.isna().any().any()):
            print('composite_signal contiene dei Nan')

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
                                          linear_filter_window_size_multiplier = 0.7,
                                          period_related_rebuild_range = True,
                                          period_related_rebuild_multiplier = 2.5,
                                          best_fit_start_back_period = None):

        last_date_index = data.index.get_loc(last_date)

        fitness = 0
        count = 0

        for index in range(periods_number):
            rel_pos  = last_date_index - periods_number + index
            current_date = data.index[rel_pos]


            print(f'Day n. {index}, calling CDC_vs_detrended_correlation()')
            print(f'\t periods_number {periods_number}, num_samples {num_samples}, final_kept_n_dominant_circles {final_kept_n_dominant_circles}, hp_filter_lambda {hp_filter_lambda}, period_related_rebuild_range {period_related_rebuild_range},  period_related_rebuild_multiplier {period_related_rebuild_multiplier}, linear_filter_window_size_multiplier {linear_filter_window_size_multiplier}')
            
            temp_fitness = self.CDC_vs_detrended_correlation(
                                                            current_date = current_date,
                                                            data = data,
                                                            num_samples= num_samples,
                                                            final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                            min_period = min_period,
                                                            max_period = max_period,
                                                            hp_filter_lambda = hp_filter_lambda,                
                                                            opt_algo_type = opt_algo_type, 
                                                            detrend_type = detrend_type,
                                                            linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                            period_related_rebuild_range = period_related_rebuild_range,
                                                            period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                                                            best_fit_start_back_period = best_fit_start_back_period
                                                          )
            
            print(f'\ttemp_fitness: {temp_fitness}')

            fitness += temp_fitness

            if(temp_fitness > 0):
                count += 1
                

        if(count > 0):
            fitness = fitness / count

        print(f'\t--> Final fitness: {fitness} for {index} days')

        return fitness



    def trade_predicted_dominant_cicles_peaks_sum(self,
                                                  data,
                                                  last_date = '2022-08-26',
                                                  periods_number = 200,
                                                  num_samples= 76,
                                                  final_kept_n_dominant_circles=2,
                                                  min_period = 10,
                                                  max_period = 18,
                                                  hp_filter_lambda = 17):


        value_sum = 0
        max_loss = 0
        max_cumulative_loss = 0
        previous_PL = 0
        temp_max_cumulative_loss = 0
        last_date_index = data.index.get_loc(last_date)
        PL = 0
        profits_sum = 0
        profits_count = 0
        losses_sum = 0
        losses_count = 0

        for index in range(periods_number):
            rel_pos  = last_date_index - periods_number + index
            current_date = data.index[rel_pos]


            PL = self.trade_predicted_dominant_cicles_peaks(
                                                      current_date = current_date,
                                                      data = data,
                                                      num_samples= num_samples,
                                                      final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                      min_period = min_period,
                                                      max_period = max_period,
                                                      hp_filter_lambda = hp_filter_lambda
                                                    )

            if((PL != None) and (not pd.isna(PL))):
                value_sum += PL

                if(PL < max_loss):
                    max_loss = PL

                if(PL < 0):
                    temp_max_cumulative_loss += PL
                    losses_sum += PL
                    losses_count += 1

                if(PL > 0):
                    temp_max_cumulative_loss = 0
                    profits_sum += PL
                    profits_count += 1

            if(temp_max_cumulative_loss < max_cumulative_loss):
                max_cumulative_loss = temp_max_cumulative_loss

        return value_sum, max_loss, max_cumulative_loss, profits_sum, profits_count, losses_sum, losses_count



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
                
        
        

    def MultiAn_initializeIndividual(self):

        random_init = []

#         print("in MultiAn_initializeIndividual")
#         display(self.MultiAn_dominant_cycles_df)

        for _ in range(len(self.MultiAn_dominant_cycles_df)):
            # print('\tself.MultiAn_detrended_max: ' + str(self.MultiAn_detrended_max))
            random_init.append( random.randint(0, int(self.MultiAn_detrended_max)) )
            
#         print(f"Initialized individual: {random_init}")

        return random_init


    def MultiAn_evaluateFitness(self, individual, return_list_type = True):
        
#         print('in MultiAn_evaluateFitness')
        
        # Verifica che individual e MultiAn_reference_detrended_data siano array NumPy
        if not isinstance(individual, np.ndarray):
            individual = np.array(individual, dtype=np.float64)

        if not isinstance(self.MultiAn_reference_detrended_data, np.ndarray):
            self.MultiAn_reference_detrended_data = np.array(self.MultiAn_reference_detrended_data, dtype=np.float64)
            
#         print('1')

        # Verifica che len_series sia un intero
        len_series = len(self.MultiAn_reference_detrended_data)

        # Verifica che period_related_rebuild_multiplier sia un double
        period_related_rebuild_multiplier = float(self.period_related_rebuild_multiplier)

        # Verifica che fitness_type sia una stringa
        fitness_type = str(self.MultiAn_fitness_type)
        
#         print('2')
        
         # Converte il dataframe in una lista di dizionari
        if isinstance(self.MultiAn_dominant_cycles_df, pd.DataFrame):
            dominant_cycles_list = self.MultiAn_dominant_cycles_df.to_dict(orient="records")
        else:
            raise TypeError("MultiAn_dominant_cycles_df deve essere un DataFrame")

        assert isinstance(dominant_cycles_list, list)
        
#         print('3')
        
        for cycle in dominant_cycles_list:
            assert isinstance(cycle, dict)
            assert 'peak_frequencies' in cycle
            assert 'peak_phases' in cycle
            assert 'peak_periods' in cycle
            assert 'start_rebuilt_signal_index' in cycle
            
#         print('4')
            
        assert individual.dtype == np.float64
        assert self.MultiAn_reference_detrended_data.dtype == np.float64
        
#         print('Tutti i tipi dati sono corretti')
        
#         print(f'individual: {individual}')
#         print(f'self.MultiAn_reference_detrended_data: {self.MultiAn_reference_detrended_data[0:10]}')
#         print(f'dominant_cycles_list: {dominant_cycles_list}')
#         print(f'len(self.MultiAn_reference_detrended_data): {len(self.MultiAn_reference_detrended_data)}')
#         print(f'self.best_fit_start_back_period: {self.best_fit_start_back_period}')
#         print(f'self.period_related_rebuild_range: {self.period_related_rebuild_range}')
#         print(f'self.period_related_rebuild_multiplier: {self.period_related_rebuild_multiplier}')
#         print(f'self.period_related_rebuild_multiplier: {self.period_related_rebuild_multiplier}')
#         print(f'self.MultiAn_fitness_type: {self.MultiAn_fitness_type}')
#         print(f'return_list_type: {return_list_type}')
        
        if(return_list_type):
            return_list_type_bool = 1
        else:
            return_list_type_bool = 0
            
        if(self.best_fit_start_back_period is None):
            best_fit_start_back_period_int = 0
        else:
            best_fit_start_back_period_int = self.best_fit_start_back_period
            
        if(self.period_related_rebuild_range is False):
            period_related_rebuild_range_int = 0
        else:
            period_related_rebuild_range_int = 1
            
      
        return evaluate_fitness(
                                individual,
                                self.MultiAn_reference_detrended_data,
                                dominant_cycles_list,
                                # self.MultiAn_dominant_cycles_df,
#                                 self.scaler,
                                len(self.MultiAn_reference_detrended_data),
                                best_fit_start_back_period_int,
                                period_related_rebuild_range_int,
                                self.period_related_rebuild_multiplier,
                                self.MultiAn_fitness_type,
                                return_list_type_bool
                               )

#     &individual_obj, &reference_data_obj, &dominant_cycles_obj,
#                           &scaler_obj, &len_series, &best_fit_start_back_period, &period_related_rebuild_range,
#                           &period_related_rebuild_multiplier, &fitness_type, &return_list_type
    
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
        #    len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period

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
                    print('period_related_rebuild_index < start_rebuilt_signal_index')
                    
                temp_circle_signal_2 = [0.0] * len(temp_circle_signal)

                # Copy values from start_evaluation_index onwards
                temp_circle_signal_2[period_related_rebuild_index:] = temp_circle_signal[period_related_rebuild_index:]
                temp_circle_signal = temp_circle_signal_2

            composite_dominant_cycle_signal += temp_circle_signal
    
        if(self.best_fit_start_back_period is None or self.best_fit_start_back_period == 0):
            
            #  it takes the max index, meaning it evaluates the lowest period used for the Goertzel transform
            max_pos = self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].max()
#             print(f'start_rebuilt_signal_index.max() = ')
#             print(self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].max())
            
            
            
        else:
            # it takes a given index
            max_pos = len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period
#             print(f'len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period =')            
#             print(len(self.MultiAn_reference_detrended_data) - self.best_fit_start_back_period)
            
        
        if(self.MultiAn_fitness_type == "mse"):

            fitness = mean_squared_error(self.MultiAn_reference_detrended_data[max_pos:], scaler.fit_transform( composite_dominant_cycle_signal[max_pos:].values.reshape(-1, 1)).flatten() * 100) # composite_dominant_cycle_signal[max_pos:])

        if(self.MultiAn_fitness_type == "just_mins_maxes"):

            mins = argrelextrema(composite_dominant_cycle_signal[max_pos:].values, np.less)[0]
            maxes = argrelextrema(composite_dominant_cycle_signal[max_pos:].values, np.greater)[0]
            peaks_indexes = np.concatenate([mins, maxes])

            fitness = mean_squared_error(self.MultiAn_reference_detrended_data[max_pos:].iloc[peaks_indexes], scaler.fit_transform( composite_dominant_cycle_signal[max_pos:].iloc[peaks_indexes].values.reshape(-1, 1)).flatten() * 100)
            
#         print(f'Evaluate fitness: {fitness}')

        if(return_list_type == True):
            return (fitness, )

        else:
            return fitness



    def MultiAn_cyclesAlignKPI(self, signals, start_position, weights = None, periods = None):

        last_position = len(signals) #- 1
#         kpi_series = pd.Series([0] * start_position)
#         weigthed_kpi_series = pd.Series([0] * start_position)

        # pre-populate first positions not used for the analysis with zeros
        kpi_series = pd.Series([0] * start_position, dtype=np.int64)
        weigthed_kpi_series = pd.Series([0] * start_position, dtype=np.int64)
        
#         print(f'start_position {start_position}, last_position {last_position}')
#         print(f'len kpi_series {len(kpi_series)}')
#         print(f'signal columns: {signals.columns}')       
        
        peaks_min_df = {} #pd.DataFrame()
        peaks_max_df = {} # pd.DataFrame()
        
        # peaks series creation
        for column in signals.columns:

            peaks_min_df[column] = argrelmin(signals[column].values)[0]
            peaks_max_df[column] = argrelmax(signals[column].values)[0]


        if(weights != None):
            weigths_sum = sum(weights)

        else:
            weigths_sum = 0


        for position in range(start_position, last_position):

            # print('\tposition: ' + str(position))
            kpi = 0
            weigthed_kpi = 0

            # print("\n")

            weigths_index = 0

            for column in signals.columns:

#                 print('\ncolumn: ' + str(column) )
                # Trova gli indici di tutti i minimi e massimi relativi
#                 peaks_min = argrelmin(signals[column].values)[0]
#                 peaks_max = argrelmax(signals[column].values)[0]

                peaks_min = peaks_min_df[column]
                peaks_max = peaks_max_df[column]
                
#                 print(f'peaks_min: {peaks_min}')
#                 print(f'peaks_max: {peaks_max}')

                # Trova i minimi e massimi immediatamente precedenti a position
                peaks_min_before = [peak for peak in peaks_min if peak < position]
                peaks_max_before = [peak for peak in peaks_max if peak < position]

                # Trova i minimi e massimi immediatamente successivi a position
                peaks_min_after = [peak for peak in peaks_min if peak > position]
                peaks_max_after = [peak for peak in peaks_max if peak > position]

                # Trova il minimo immediatamente precedente più vicino a position
                previous_peak_index_min = max(peaks_min_before) if peaks_min_before else np.nan

                # Trova il massimo immediatamente precedente più vicino a position
                previous_peak_index_max = max(peaks_max_before) if peaks_max_before else np.nan

                # Trova il minimo immediatamente successivo più vicino a position
                next_peak_index_min = min(peaks_min_after) if peaks_min_after else np.nan

                # Trova il massimo immediatamente successivo più vicino a position
                next_peak_index_max = min(peaks_max_after) if peaks_max_after else np.nan

                # Scegli l'indice corretto in base alla distanza dalla posizione corrente
                if not pd.isnull(previous_peak_index_max) and not pd.isnull(previous_peak_index_min):

                    if(position - previous_peak_index_max) < (position - previous_peak_index_min):
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
                    
#                 print(f'\tprevious_peak_index_max {previous_peak_index_max}')
#                 print(f'\tprevious_peak_index_min {previous_peak_index_min}')
#                 print(f'\tprevious_peak_index {previous_peak_index}')
#                 print(f'\tprevious_peak_type {previous_peak_type}')


                if not pd.isnull(previous_peak_index) and not pd.isnull(next_peak_index):
                    # Calcola la lunghezza del percorso tra i due picchi.
                    path_len = next_peak_index - previous_peak_index

                    # Calcola la lunghezza del percorso che rimane.
                    remain_len = next_peak_index - position

                    # Calcola la percentuale di percorso che rimane.
                    percentage = remain_len / path_len


                    # Aggiungi la percentuale al KPI finale.
                    if(previous_peak_type == 'min'):
                        kpi -= percentage

                    if(previous_peak_type == 'max'):
                        kpi += percentage

                    if(weights != None):

                        if(periods is not None):

                            if(previous_peak_type == 'min'):
                                weigthed_kpi -= percentage * weights[weigths_index] / periods[weigths_index]

                            if(previous_peak_type == 'max'):
                                weigthed_kpi += percentage * weights[weigths_index] / periods[weigths_index]

                        else:

                            if(previous_peak_type == 'min'):
                                weigthed_kpi -= percentage * weights[weigths_index] / weigths_sum

                            if(previous_peak_type == 'max'):
                                weigthed_kpi += percentage * weights[weigths_index] / weigths_sum

                weigths_index += 1
                
#                 print(f'\tweigths_index {weigths_index}')

#             kpi_series = kpi_series.append(pd.Series([kpi]), ignore_index=True)
#             weigthed_kpi_series = weigthed_kpi_series.append(pd.Series([weigthed_kpi]), ignore_index=True)
        
            kpi_series = pd.concat([kpi_series, pd.Series([kpi])], ignore_index=True)
            weigthed_kpi_series = pd.concat([weigthed_kpi_series, pd.Series([weigthed_kpi])], ignore_index=True)


#             has_nan_kpi_series = kpi_series.isna().any()
#             # Stampa il risultato
#             print(f"Ci sono valori NaN in kpi_series? {has_nan_kpi_series}")

#             # Controlla la presenza di NaN nella nuova serie pd.Series([kpi])
#             has_nan_kpi = pd.Series([kpi]).isna().any()
#             # Stampa il risultato
#             print(f"Ci sono valori NaN in pd.Series([kpi])? {has_nan_kpi}")

#             # Stampa i tipi di dato delle serie
#             print(f"Tipo di dato di kpi_series: {kpi_series.dtype}")
#             print(f"Tipo di dato di pd.Series([kpi]): {pd.Series([kpi]).dtype}")

#             # Concatena le serie
#             kpi_series = pd.concat([kpi_series, pd.Series([kpi])], ignore_index=True)
#             weigthed_kpi_series = pd.concat([weigthed_kpi_series, pd.Series([weigthed_kpi])], ignore_index=True)

#             # Stampa i tipi di dato dopo la concatenazione
#             print(f"Tipo di dato di kpi_series dopo la concatenazione: {kpi_series.dtype}")
#             print(f"Tipo di dato di weigthed_kpi_series dopo la concatenazione: {weigthed_kpi_series.dtype}")

#         print('kpi_series: ' + str(kpi_series.tail(50)))
#         print('weigthed_kpi_series: ' + str(weigthed_kpi_series.tail(50)))


        return kpi_series, weigthed_kpi_series



    # Fitness function
    def genOpt_evaluateMSEFitness(self, individual):

        data = self.data
        last_date = self.genOpt_last_date
#         logarithmic_sequence = self.genOpt_logarithmic_sequence
        periods_number = self.genOpt_periods_number
        period_related_rebuild_multiplier = 1
        linear_filter_window_size_multiplier = 1
        hp_filter_lambda = 1

        if(self.detrend_type == 'hp_filter'):
            
            if(self.period_related_rebuild_range == True):                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda, period_related_rebuild_multiplier  = individual
                
                if((period_related_rebuild_multiplier < 2) or (period_related_rebuild_multiplier > 6)):
                    print('Constraint violation, (period_related_rebuild_multiplier < 1) or (period_related_rebuild_multiplier > 5)')
                    return (1e9,)  # Vincolo violato
                
            else:                                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda  = individual            
            
            
            if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
                print('Constraint violation, hp_filter_lambda < 1 or hp_filter_lambda > 2e9')
                return (1e9,)  # Vincolo violato
            
            
            
        elif(self.detrend_type == 'linear'):
            
            if(self.period_related_rebuild_range == True):                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, linear_filter_window_size_multiplier, period_related_rebuild_multiplier = individual
                
                if((period_related_rebuild_multiplier < 1) or (period_related_rebuild_multiplier > 5)):
                    print('Constraint violation, (period_related_rebuild_multiplier < 1) or (period_related_rebuild_multiplier > 5)')
                    return (1e9,)  # Vincolo violato
                
            else:                
                num_samples, final_kept_n_dominant_circles, min_period, max_period, linear_filter_window_size_multiplier  = individual                
                
            if((linear_filter_window_size_multiplier < 0.5) or (linear_filter_window_size_multiplier > 2)):
                    print('Constraint violation, (linear_filter_window_size_multiplier < 0.5) or (linear_filter_window_size_multiplier > 2)')
                    return (1e9,)  # Vincolo violato

        # Constraints definition
        if num_samples < max_period*2:
            print('Constraint violation, num_samples < max_period*2')
            return (1e9,)  # Vincolo violato
        if num_samples > 5000 or num_samples < 30:
            print('Constraint violation, num_samples > 5000 or num_samples < 50')
            return (1e9,)  # Vincolo violatoo
        if final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1:
            print('Constraint violation, final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1')
            return (1e9,)  # Vincolo violato
        if min_period < 1 or min_period > 1024:
            print('Constraint violation, min_period < 1 or min_period > 1024')
            return (1e9,)  # Vincolo violato
        if max_period < 7 or max_period > 1024:
            print('Constraint violation, max_period < 10 or max_period > 1024')
            return (1e9,)  # Vincolo violato
        if min_period + 2 > max_period:
            print('Constraint violation, min_period + 2 > max_period')
            return (1e9,)  # Vincolo violato
        if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
            print('Constraint violation, hp_filter_lambda < 1 or hp_filter_lambda > 2e9')
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
                linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                period_related_rebuild_range = self.period_related_rebuild_range,
                period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                best_fit_start_back_period = self.best_fit_start_back_period
            )
            
            print(f'Par opt, fitness: {fitness}')


        except Exception as e:

            print(f"An exception occurred in genOpt_evaluateMSEFitness() calling CDC_vs_detrended_correlation_sum(): {e}")
            print("Traceback:")
            traceback.print_exc()
            print(f'last_date: {last_date}')
            print(f'periods_number: {periods_number}')
            print(f'final_kept_n_dominant_circles: {final_kept_n_dominant_circles}')
            print(f'min_period: {min_period}')
            print(f'max_period: {max_period}')
            print(f'best_fit_start_back_period: {self.best_fit_start_back_period}')
            
            if(self.detrend_type == 'hp_filter'):
                print(f'hp_filter_lambda: {hp_filter_lambda}')
            
            if(self.detrend_type == 'linear'):
                print(f'linear_filter_window_size_multiplier: {linear_filter_window_size_multiplier}')
            
            
            if(self.period_related_rebuild_range == True):
                print(f'period_related_rebuild_multiplier: {period_related_rebuild_multiplier}')

#             sys.exit()

            return (1e9, )

        else:

#             print(f'fitness: {fitness}')
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
            print('Constraint violation, num_samples > 1200 or num_samples < 50')
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1:
            print('Constraint violation, final_kept_n_dominant_circles > 15 or final_kept_n_dominant_circles < 1')
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if min_period < 1 or min_period > 200:
            print('Constraint violation, min_period < 1 or min_period > 200')
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if max_period < 7 or max_period > 400:
            print('Constraint violation, max_period < 10 or max_period > 400')
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if min_period + 2 > max_period:
            print('Constraint violation, min_period + 2 > max_period')
            return (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, 1e9, -1e9, -1e9)   # Vincolo violato
        if hp_filter_lambda < 1 or hp_filter_lambda > 2e9:
            print('Constraint violation, hp_filter_lambda < 1 or hp_filter_lambda > 2e9')
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

            print(f"genOpt_evaluateFitness, pos 1: an exception occurred: {e}")
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
#                                         linear_filter_window_size_multiplier = 1,
                                        period_related_rebuild_range = False,
#                                         period_related_rebuild_multiplier = 2.5,
                                        population_n = 100, # Population number
                                        NGEN = 10,  # Generations number
                                        CXPB = 0.7,  # Crossover probability
                                        MUTPB = 0.35,  # Mutation probability
                                        fitness_function = 'trading_pl', # 'trading_pl', 'mse'
                                        enabled_multiprocessing = True,
                                        time_tracking = False,
                                        print_activity_remarks = False
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
        
        self.print_activity_remarks = print_activity_remarks
        
        # if not specified use global class folder path
        if(folder_path is None):
            folder_path = self.data_storage_path 
        
        # if not specified use default file name
        if(file_name is None):
            file_name = self.ticker + "_cyclces_analysis_hypeparmeters_optimization"

        
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

        
        
#         [
#             'run_start_datetime',
#             'run_end_datetime',
#             'ticker_symbol',
#             'optimization_label',
#             'generation_number',            
#             'fitness_function',
#             'detrend_type',
#             'opt_algo_type',
# #             'linear_filter_window_size_multiplier', # <-- missing, from best individual
#             'period_related_rebuild_range',
# #             'period_related_rebuild_multiplier', # <-- missing, from best individual           
#             'opt_pars_last_date',
#             'opt_pars_optimization_label',
#             'opt_pars_num_samples_min',
#             'opt_pars_num_samples_max',
#             'opt_pars_final_kept_n_dominant_circles_min',
#             'opt_pars_final_kept_n_dominant_circles_max',
#             'opt_pars_min_period_min',
#             'opt_pars_min_period_max',
#             'opt_pars_max_period_min',
#             'opt_pars_max_period_max',
#             'opt_pars_hp_filter_lambda_min',
#             'opt_pars_hp_filter_lambda_max',
#             'opt_pars_hp_filter_lambda_n',
#             'opt_pars_periods_number',
#             'linear_filter_window_size_multiplier', 
#             'period_related_rebuild_range', 
#             'period_related_rebuild_multiplier', 
#             'best_fit_start_back_period', 
#             'opt_pars_population_n',
#             'opt_pars_NGEN',
#             'opt_pars_CXPB',
#             'opt_pars_MUTPB',
#             'best_individual_num_samples',
#             'best_individual_final_kept_n_dominant_circles',
#             'best_individual_min_period',
#             'best_individual_max_period',
#             'best_individual_linear_filter_window_size_multiplier', 
#             'best_individual_period_related_rebuild_multiplier',   
#             'best_individual_hp_filter_lambda',
#             'best_fitness_value_sum',
#             'best_fitness_max_loss',
#             'best_fitness_max_cumulative_loss',
#             'best_fitness_profits_sum',
#             'best_fitness_profits_count',
#             'best_fitness_losses_sum',
#             'best_fitness_losses_count',
#             'best_fitness_count_profits_vs_losses',
#             'best_fitness_profits_vs_losses'
#         ]

        results_history = pd.DataFrame(columns=column_names)

#         display(results_history)

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
        self.time_tracking = time_tracking



        # CREATION OF THE SEQUENCES OF VALUES TO BE TESTED ACCORDING TO THE TYPE OF DETRENDING
        
        if(self.period_related_rebuild_range == True): 
#             self.period_related_rebuild_multiplier_sequence = np.arange(2, 6.05, 0.05).tolist()
            
            print('Creating period_related_rebuild_range')
        
            start = Decimal('2.0')
            stop = Decimal('6.05')
            step = Decimal('0.05')

            self.period_related_rebuild_multiplier_sequence = []
            current = start
            while current <= stop:
                self.period_related_rebuild_multiplier_sequence.append(float(current))
                current += step
    
            print(f'period_related_rebuild_multiplier_sequence: {self.period_related_rebuild_multiplier_sequence}')
            
        if(self.detrend_type == 'hp_filter'):
            logarithmic_sequence = np.logspace(np.log10(hp_filter_lambda_min), np.log10(hp_filter_lambda_max), num=hp_filter_lambda_n, dtype=int)
            logarithmic_sequence = np.unique(logarithmic_sequence)
            self.genOpt_logarithmic_sequence = logarithmic_sequence
            
            # Variable to the specific component period rebuild range
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
                
            # Default rebuild range (equal to the frequancy transfor periods)
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
                        
            # Variable to the specific component period rebuild range
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
                
            # Default rebuild range (equal to the frequancy transform periods)    
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



        # CREATOR DEPENDING ON THE FITNESS FUNCTION TYPE 
        if(fitness_function == 'trading_pl'): # 'trading_pl'
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 0.4, 0.4, 1, 1, 0.6, -0.6, 1, 1)) # loss are always negative so they must be maximized
        else: # 'mse'
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, ))

        # Crea il creator per un individuo (una lista di parametri)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Inizializzazione della popolazione e delle operazioni genetiche
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
            print(f'multiprocessing.get_start_method = {multiprocessing.get_start_method()}')

            # Set the multiprocessing start method to 'spawn'
            if multiprocessing.get_start_method() != 'spawn':
                print('Setting Spawn method')
                multiprocessing.set_start_method('spawn')

            cpu_count = multiprocessing.cpu_count()
            print(f"CPU count: {cpu_count}")

            # Enable multiprocessing
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
        
        else:
            
            print('Multiprocessing disabled')


        # Create the initial population
        population = toolbox.population(n=population_n)

#         folder_path = self.data_storage_path 
        
#         if(file_name is None):
#             file_name = self.ticker + "_cyclces_analysis_hypeparmeters_optimization"
        
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
                print("genOpt_cycleParsGenOptimization, generation number: " + str(gen+1) + "/" + str(NGEN)+ ", population number " + str(count) + "/" + str(population_n))
                sys.stdout.flush()
                
            population = toolbox.select(offspring, k=len(population))

            best_individual = tools.selBest(population, k=1)[0]
            best_fitness = best_individual.fitness.values
            
            print("-------------------------------------------------------------------")
            print(f'best_fitness for multirange pars optimization {best_fitness}')
            print(f'best_individual for multirange pars optimization {best_individual}')
            print("-------------------------------------------------------------------")
            
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
            sys.stdout.flush()
            
            self.save_dataframe(dataframe = results_history, folder_path = folder_path, file_name = file_name)

            if(self.output_clearing == True):
                clear_output(wait=True)
                
            display(results_history)
            sys.stdout.flush()


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

        print("Final results")
        
        if(self.output_clearing == True):
            clear_output(wait=True)
            
        display(final_history)



        return final_history



    def save_dataframe(self,
                       dataframe,
                       folder_path,
                       file_name,
                       update_column = False,
                       update_column_name = None,
                       update_column_value = None,
                       filter_column_name = None,
                       filter_column_value = None):

        # Remove special characters from folder name, keep the follow
        clean_folder_name = re.sub(r'[^a-zA-Z0-9_\-\s\:\\]', '', folder_path)

        # Remove special characters from file name, keep the follow
        clean_file_name = re.sub(r'[^a-zA-Z0-9_\-\s\=]', '', file_name)
        
        
#         print(f"DataFrame going to: {clean_folder_name}")
#         print(f"file name: {clean_file_name}")

        # Create the folder if it doesn't exist
        folder_path = os.path.join(os.getcwd(), clean_folder_name)
#         print(f'Object filepath: {folder_path}')
        os.makedirs(folder_path, exist_ok=True)

        # Check if the file already exists
        file_path = os.path.join(folder_path, f"{clean_file_name}.csv")

        if os.path.isfile(file_path):
            # If the file exists, load the existing DataFrame
            existing_dataframe = pd.read_csv(file_path)
            
            existing_dataframe.reset_index(drop=True, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            
#             display(existing_dataframe)
#             display(dataframe)

            # Concatenate the existing DataFrame with the new data
            combined_dataframe = pd.concat([existing_dataframe, dataframe], ignore_index=True)
        else:
            # If the file doesn't exist, use the new DataFrame
            combined_dataframe = dataframe

        if(update_column == True and update_column_name != None and update_column_value != None):

            if(filter_column_name == None and filter_column_value == None):
                combined_dataframe[update_column_name] = update_column_value

            elif(filter_column_name != None and filter_column_value != None):

                # print('update_column: ' + update_column)
                # print('update_column_name: ' + update_column_name)
                # print('update_column_value: ' + update_column_value)
                # print('filter_column_name: ' + filter_column_name)
                # print('filter_column_value: ' + filter_column_value)

                combined_dataframe.loc[combined_dataframe[filter_column_name] == filter_column_value, update_column_name] = update_column_value

        # Save the combined DataFrame to CSV
        combined_dataframe.to_csv(file_path, index=False)


        return combined_dataframe






    def min_max_analysis(self,
                         data,
                         current_time_idx,
                         suffix_col_name,
                         N_elements = 10,
                         delta_comparison_serie = None,
                         comparison_serie_name = ''):

#         cdc_min_max_indices = np.concatenate((argrelextrema(data.values, np.less)[0], argrelextrema(data.values, np.greater)[0]))
        cdc_min_max_indices = np.concatenate((argrelextrema(data.values, np.less, order=1)[0], argrelextrema(data.values, np.greater, order=1)[0]))
        cdc_min_max_indices.sort()


        # Trovare gli indici dei minimi/massimi precedenti e successivi alla data corrente
        indices_before = cdc_min_max_indices[cdc_min_max_indices < current_time_idx][-N_elements:] #[:10]
        indices_after = cdc_min_max_indices[cdc_min_max_indices >= current_time_idx][:N_elements] #[-10:]

        # Unire gli indici
        indices_before_after = np.concatenate((indices_before, indices_after))
        dates = data.iloc[indices_before_after].index


        data_dict = {}
        previous_cdc_value = None

        for i, idx in enumerate(indices_before_after):

            abs_label_idx = abs(i-N_elements)

            col_prefix = f'{suffix_col_name}+{abs_label_idx+1}' if idx >= current_time_idx else f'{suffix_col_name}-{abs_label_idx}'

            cdc_type = 1 if data[idx] > data[idx - 1] and data[idx] > data[idx + 1] else -1
            cdc_value = data[idx]

            data_dict[f'{col_prefix}_idx_delta'] = idx - current_time_idx
            data_dict[f'{col_prefix}_type'] = cdc_type
            data_dict[f'{col_prefix}_value'] = cdc_value

            if(delta_comparison_serie is not None) :
                
                # Importance: this could be meaningful to understand the degree of error forecast between the dominant cycles and the
                #             original signal
                if(idx <= current_time_idx):
                    cdc_orig_delta = cdc_value - delta_comparison_serie[idx] #scaled_signals['scaled_detrended'][idx]
                    data_dict[f'{col_prefix}_{comparison_serie_name}_error'] = cdc_orig_delta
                
                # Importance: a major delta between previous could be a sign of a higher trend reveral probability; a minor one
                #             coudl be sign of just noise
                if(previous_cdc_value is not None): 
                    data_dict[f'{col_prefix}_{comparison_serie_name}_contiguos_peaks_delta'] = abs(cdc_value - previous_cdc_value)

    
            previous_cdc_value = cdc_value

        # Aggiungere il CDC_trend
        cdc_type = data_dict[suffix_col_name + '-1_type'] if suffix_col_name+'-1_type' in data_dict else np.nan
        cdc_trend = 1 if cdc_type == -1 else -1

        data_dict[suffix_col_name + 'CDC_trend'] = cdc_trend
        data_df = pd.DataFrame([data_dict])

#         print('current_time_idx: ' + str(current_time_idx))


        return data_df


    def min_max_analysis_concatenated_dataframe(self, #cyPredict_ist,
                                                data_column_name,
                                                current_date,
                                                periods_pars,
                                                time_zone = 'America/New_York',
                                                pars_from_opt_file = False,
                                                files_path_name = None,
                                                population_n = 10,
                                                CXPB = 0.7,
                                                MUTPB = 0.3,
                                                NGEN = 400,
                                                MultiAn_fitness_type = "mse",
                                                MultiAn_fitness_type_svg_smoothed = False,
                                                MultiAn_fitness_type_svg_filter = 4,
                                                reference_detrended_data = "less_detrended", # longest less_detrended
                                                bb_delta_fixed_periods = [8, 16],
                                                bb_delta_sg_filter_window = None,
                                                RSI_cycles_analysis_type = 'original_RSI', #'SG_smooted_RSI' # 'original_RSI',
                                                opt_algo_type = 'genetic', # 'genetic', 'tpe', 'atpe'
                                                detrend_type = 'hp_filter', #hp_filter linear
                                                linear_filter_window_size_multiplier = 1.85,
                                                period_related_rebuild_range = False,
                                                period_related_rebuild_multiplier = 1.2,
                                                CDC_bb_analysis = False,
                                                CDC_RSI_analysis = False,
                                                CDC_MACD_analysis = False,
                                                show_charts = False,
                                                N_elements_prices_CDC = 6,
                                                N_elements_goertzel_CDC = 3,
                                                N_elements_alignmentsKPI_CDC = 10,
                                                N_elements_weigthed_alignmentsKPI_CDCC = 10,
                                                enabled_multiprocessing = True
                                                ):
        
        


        datetime_df = pd.DataFrame()
        best_fitness_value_df = pd.DataFrame()
        
        

        elaborated_data_df, signals_results_df, composite_signal, configurations, bb_delta, cdc_rsi, index_of_max_time_for_cd, scaled_signals, best_fitness_value = self.multiperiod_analysis(
                                                data_column_name = data_column_name,
                                                current_date = current_date, #'2024-01-18',
                                                periods_pars = periods_pars,
                                                time_zone = time_zone,
                                                pars_from_opt_file = pars_from_opt_file,
                                                files_path_name = files_path_name,
                                                population_n = population_n,
                                                CXPB = CXPB,
                                                MUTPB = MUTPB,
                                                NGEN = NGEN,
                                                MultiAn_fitness_type = MultiAn_fitness_type,
                                                MultiAn_fitness_type_svg_smoothed = MultiAn_fitness_type_svg_smoothed,
                                                MultiAn_fitness_type_svg_filter =MultiAn_fitness_type_svg_filter,
                                                reference_detrended_data = reference_detrended_data, # longest less_detrended
                                                bb_delta_fixed_periods = bb_delta_fixed_periods,
                                                bb_delta_sg_filter_window = bb_delta_sg_filter_window,
                                                RSI_cycles_analysis_type = RSI_cycles_analysis_type, #'SG_smooted_RSI' # 'original_RSI',
                                                opt_algo_type = opt_algo_type, # 'genetic', 'tpe', 'atpe'
                                                detrend_type = detrend_type, #hp_filter linear
                                                linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                period_related_rebuild_range = period_related_rebuild_range,
                                                period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                                                CDC_bb_analysis = CDC_bb_analysis,
                                                CDC_RSI_analysis = CDC_RSI_analysis,
                                                CDC_MACD_analysis = CDC_MACD_analysis,
                                                show_charts = show_charts,
                                                enabled_multiprocessing = True
                                            )
        
#         print(f'\nCurrent date: {current_date}')
#         print(f"\tPosition in composite_signal['composite_signal']: {composite_signal['composite_signal'].index.get_loc(current_date)}")      
#         print(f'\tindex_of_max_time_for_cd: {index_of_max_time_for_cd}')
#         print(f"\tlen of composite_signal['composite_signal']: {len(composite_signal['composite_signal'])}")
#         print(f"\tlen of scaled_signals['scaled_composite_signal']: {len(scaled_signals['scaled_composite_signal'])}")

        datetime_df['datetime'] = [current_date]
        best_fitness_value_df['best_fitness_value'] = [best_fitness_value]

        min_max_prices_CDC_analysis = self.min_max_analysis(data = scaled_signals['scaled_composite_signal'],
                                                            delta_comparison_serie = scaled_signals['scaled_detrended'],
                                                            comparison_serie_name = 'detrended_prices',
                                                            current_time_idx = index_of_max_time_for_cd,
                                                            suffix_col_name = 'prices_CDC_min_max_',
                                                            N_elements = N_elements_prices_CDC)

        min_max_goertzel_CDC_analysis = self.min_max_analysis(data = pd.Series( savgol_filter(scaled_signals['scaled_goertzel_composite_signal'] , 30, 2) ),
                                                              delta_comparison_serie = scaled_signals['scaled_detrended'],
                                                              comparison_serie_name = 'detrended_prices',
                                                              current_time_idx = index_of_max_time_for_cd,
                                                              suffix_col_name = 'goertzel_CDC_min_max_',
                                                              N_elements = N_elements_goertzel_CDC)

        min_max_alignmentsKPI_CDC_analysis = self.min_max_analysis(data = scaled_signals['scaled_alignmentsKPI'],
                                                                  #  scaled_detrended = scaled_signals['scaled_detrended'],
                                                                   current_time_idx = index_of_max_time_for_cd,
                                                                   suffix_col_name = 'scaled_alignmentsKPI_',
                                                                   N_elements = N_elements_alignmentsKPI_CDC)

        min_max_weigthed_alignmentsKPI_CDC_analysis = self.min_max_analysis(data = scaled_signals['scaled_weigthed_alignmentsKPI'],
                                                                            # scaled_detrended = scaled_signals['scaled_detrended'],
                                                                            current_time_idx = index_of_max_time_for_cd,
                                                                            suffix_col_name = 'weigthed_alignmentsKPI_',
                                                                            N_elements = N_elements_weigthed_alignmentsKPI_CDCC)

        base_data = pd.DataFrame()
        base_data = pd.DataFrame([self.data.iloc[index_of_max_time_for_cd][['Open', 'Low', 'High', 'Close', 'Volume']].values], columns=['Open', 'Low', 'High', 'Close', 'Volume'])
        # pd.DataFrame(self.data.iloc[index_of_max_time_for_cd][['Open', 'Low', 'High', 'Close', 'Volume']]).T # self.data.iloc[index_of_max_time_for_cd][['Open', 'Low', 'High', 'Close', 'Volume']]
        #[['Open', 'Low', 'High', 'Close', 'Volume']]

        base_data['CO'] = base_data['Close'] - base_data['Open']
        base_data['HL'] = base_data['High'] - base_data['Low']
        base_data['CL'] = base_data['Close'] - base_data['Low']
        base_data['CH'] = base_data['Close'] - base_data['High']
        base_data['HO'] = base_data['High'] - base_data['Open']
        base_data['OL'] = base_data['Open'] - base_data['Low']
#         display(base_data)

        base_data['HL_Volume_effort'] = base_data['HL'] / base_data['Volume']


        if(index_of_max_time_for_cd > 0):
            base_data['Open_delta'] = self.data.iloc[index_of_max_time_for_cd]['Open'] - self.data.iloc[index_of_max_time_for_cd - 1]['Open']
            base_data['Close_delta'] = self.data.iloc[index_of_max_time_for_cd]['Close'] - self.data.iloc[index_of_max_time_for_cd - 1]['Close']
            base_data['High_delta'] = self.data.iloc[index_of_max_time_for_cd]['High'] - self.data.iloc[index_of_max_time_for_cd - 1]['High']
            base_data['Low_delta'] = self.data.iloc[index_of_max_time_for_cd]['Low'] - self.data.iloc[index_of_max_time_for_cd - 1]['Low']
            base_data['Volume_delta'] = self.data.iloc[index_of_max_time_for_cd]['Volume'] - self.data.iloc[index_of_max_time_for_cd - 1]['Volume']

            base_data['Close_Volume_effort_delta'] = base_data['Close_delta'] / base_data['Volume_delta']

        else:
            base_data['Open_delta'] = 0
            base_data['Close_delta'] = 0
            base_data['High_delta'] = 0
            base_data['Low_delta'] = 0
            base_data['Volume_delta'] = 0

            base_data['Close_Volume_effort_delta'] = 0
            
        # TO BE ADDED:
        #    - the detrended original prices
        #    - the delta between the next 2nd, 3th, 4th, 5th min_max and the very next one. Importance: if the second one is a max 
        #      and the third that is a max too is higher than the first one, maybe it could be ignore the min in the middle since the main 
        #      trend is rising (dually if the next one is a min) --> Warning: at the turning point the next max will become the 
        #      current or previous max so it will be worth to trak anlso the preious min_max-1:
        #         - delta_next_1_min_max = min_max+1_value - min_max-1_value
        #         - delta_next_1_min_max = min_max+2_value - min_max-1_value
        #         - delta_next_1_min_max = min_max+3_value - min_max-1_value
        #         - delta_next_1_min_max = min_max+4_value - min_max-1_value
        #         - delta_next_1_min_max = min_max+5_value - min_max-1_value
        
        base_data['scaled_detrended'] = scaled_signals['scaled_detrended']


        concatenated_dataframe = pd.concat([datetime_df,
                                            best_fitness_value_df,
                                            base_data,
                                            min_max_prices_CDC_analysis,
                                            min_max_goertzel_CDC_analysis,
                                            min_max_alignmentsKPI_CDC_analysis,
                                            min_max_weigthed_alignmentsKPI_CDC_analysis], axis=1)

        return concatenated_dataframe



    def get_min_max_analysis_df(self,
                                cycles_parameters,
                                current_date,
                                lookback_periods,
                                source_type: Type[Drive],
                                retrieve_pars_from_file = False,
                                optimized_pars_filepath = None,
                                min_period = None,
                                max_period = None,
                                data_column_name = 'Close',
                                detrend_type = 'hp_filter', #hp_filter linear
                                linear_filter_window_size_multiplier = 1.85,
                                period_related_rebuild_range = False,
                                period_related_rebuild_multiplier = 1.2,
                                population_n = 10,
                                CXPB = 0.7,
                                MUTPB = 0.3,
                                NGEN = 400,
                                resume = False,
                                GoogleDriveMountPoint = '/content/drive',
                                file_path = '/My Drive',
                                file_name = '/min_max_prices_analysis.csv',
                                index_column_name = 'datetime'):
        
        file_path_name = file_path + file_name
        
        # Carica il CSV esistente solo se resume == True e il file esiste
        if resume and os.path.exists(file_path_name):
#             min_max_CDC_analysis_df = pd.read_csv(file_path_name, index_col=index_column_name, parse_dates=True)
            min_max_CDC_analysis_df = pd.read_csv(file_path_name, parse_dates=['datetime'])

            print("File CSV esistente caricato.")
        else:
            # Se il CSV non esiste o resume == False, crea un nuovo dataframe vuoto
            min_max_CDC_analysis_df = pd.DataFrame()
            print("Nessun file CSV trovato o resume == False. Creazione di un nuovo dataframe.")
            

        # Filtra i dati esistenti in self.data per la data corrente e un numero di righe precedenti pari a lookback_periods
        if pd.to_datetime(current_date) not in self.data.index:
            print(f"Data {current_date} non trovata nei dati. Uscita.")
            return min_max_CDC_analysis_df

        # Ottieni i dati necessari dal dataframe self.data
        start_idx = self.data.index.get_loc(pd.to_datetime(current_date)) - lookback_periods
        if start_idx < 0:
            start_idx = 0
        filtered_data = self.data.iloc[start_idx:self.data.index.get_loc(pd.to_datetime(current_date)) + 1]
        
#         print(f'Filtered data len {len(filtered_data)}')
        
        
#         print(f'Number downloaded asset records: {len(filtered_data)}')
        
        # Filtra solo le date che non sono già presenti in min_max_CDC_analysis_df
        if not min_max_CDC_analysis_df.empty:            

            # Assicurati che l'indice di min_max_CDC_analysis_df sia del tipo corretto
            min_max_CDC_analysis_df['datetime'] = pd.to_datetime(min_max_CDC_analysis_df['datetime'], errors='coerce')

            # Controlla le date che non sono già nel CSV
            missing_dates = filtered_data.index.difference(min_max_CDC_analysis_df['datetime'])

            # Seleziona solo le righe con le date mancanti
            filtered_data = filtered_data.loc[missing_dates]
#             print(f'Number of missing dates after further filtering: {len(filtered_data)}')

        else:
            print("Il file CSV è vuoto, quindi tutte le date sono considerate nuove.")


        # Esegui l'analisi su ogni nuova data usando min_max_analysis_concatenated_dataframe
        for date in filtered_data.index:
            # Converte la data in formato "YYYY-MM-DD"
            date_str = date.strftime('%Y-%m-%d')
            
            
            print('\n\n----------------------------------------------------------')
            print(f'min_max_analysis_concatenated_dataframe on date {date_str}')
            print('----------------------------------------------------------')
            
            if(retrieve_pars_from_file and optimized_pars_filepath is not None):
                
                cycles_parameters = self.get_most_updated_optimization_pars(optimized_pars_filepath, date_str)

                
#                 cycles_parameters = cycles_parameters[['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda']]
                
        
            if(min_period is not None):
                cycles_parameters = cycles_parameters[cycles_parameters['min_period'] >= min_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)

            if(max_period is not None):
                cycles_parameters = cycles_parameters[cycles_parameters['max_period'] <= max_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)
                    
                
            # Richiama min_max_analysis_concatenated_dataframe con i parametri corretti
            analyzed_row = self.min_max_analysis_concatenated_dataframe(
                                                  data_column_name = 'Close',
                                                  current_date = date_str, #'2024-01-18',
                                                  periods_pars = cycles_parameters,
                                                  time_zone = 'America/New_York',
                                                  pars_from_opt_file = False,
                                                  files_path_name = None,
                                                  population_n = population_n,
                                                  CXPB = CXPB,
                                                  MUTPB = MUTPB,
                                                  NGEN = NGEN,
                                                  MultiAn_fitness_type = "mse",
                                                  MultiAn_fitness_type_svg_smoothed = False,
                                                  MultiAn_fitness_type_svg_filter = 4,
                                                  reference_detrended_data = "less_detrended", # longest less_detrended
                                                  bb_delta_fixed_periods = [8, 16],
                                                  bb_delta_sg_filter_window = None,
                                                  RSI_cycles_analysis_type = 'original_RSI', #'SG_smooted_RSI' # 'original_RSI',
                                                  opt_algo_type = 'genetic_omny_frequencies',
                                                  detrend_type = detrend_type, #hp_filter linear
                                                  linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                  period_related_rebuild_range = period_related_rebuild_range,
                                                  period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                                                  CDC_bb_analysis = False,
                                                  CDC_RSI_analysis = False,
                                                  CDC_MACD_analysis = False,
                                                  show_charts = False
            )
            
            if 'datetime' in analyzed_row.columns:
                analyzed_row['datetime'] = pd.to_datetime(analyzed_row['datetime'], errors='coerce')

            # Concatena il risultato al dataframe principale
            min_max_CDC_analysis_df = pd.concat([min_max_CDC_analysis_df, analyzed_row])

       
        # Converte tutto l'indice in Timestamp per evitare conflitti tra tipi diversi
            min_max_CDC_analysis_df = min_max_CDC_analysis_df.sort_values(by='datetime')
            min_max_CDC_analysis_df = min_max_CDC_analysis_df.drop_duplicates(subset=['datetime'], keep='first')


        # Salva il dataframe risultante nel CSV, sovrascrivendo il file esistente
            min_max_CDC_analysis_df.to_csv(file_path_name, index=False)
            print(f"CSV aggiornato e salvato: {file_path_name}")

        return min_max_CDC_analysis_df
    
    
    
    def get_most_updated_optimization_pars(self, file_path, current_date = None, print_df_code = False):

        df = pd.read_csv(file_path)

        # Assicurati che la colonna analysis_reference_date sia in formato datetime
        df['analysis_reference_date'] = pd.to_datetime(df['analysis_reference_date'])
        df['best_fitness_value_sum'] = df['best_fitness_value_sum'].fillna(1e10)

        if(current_date is None):
            # Ottieni la data attuale
            current_date = datetime.now()
        elif isinstance(current_date, str):
            # Se current_date è una stringa, convertila in datetime
            current_date = pd.to_datetime(current_date)


        # Filtra df in base a detrend_type e period_related_rebuild_range
        filtered_df = df[(df['opt_period_related_rebuild_range'] == False) & (df['detrend_type'] == 'hp_filter')]

        # Iniziamo raggruppando per optimization_label
        result_list = []


        for label, group in filtered_df.groupby('optimization_label'):
            
#             print(f'Current date {current_date}')

            # Filtra solo le date che sono nel passato o uguali alla data corrente
            group = group[group['analysis_reference_date'] <= current_date]

            if not group.empty:
                # Per ogni optimization_label, calcola la differenza assoluta tra le date e la data corrente
                group['time_diff'] = (current_date - group['analysis_reference_date']).abs()
                
#                 display(group)

                # Trova l'indice della riga con il valore minimo di 'time_diff' per ciascun gruppo
                idx_min_time_diff = group['time_diff'].idxmin()
#                 display(idx_min_time_diff)

                # Filtra solo la riga con la data più vicina alla data corrente (passata)
                closest_group = group.loc[[idx_min_time_diff]]  # Note le doppie parentesi per restituire un DataFrame


                # Trova l'indice della riga con il valore minimo di 'best_fitness_value_sum' per ciascun gruppo filtrato
                idx_min_fitness = closest_group.groupby('optimization_label')['best_fitness_value_sum'].idxmin()

                # Seleziona le righe corrispondenti a questi indici e le colonne richieste
                result = df.loc[idx_min_fitness, [
                    'analysis_reference_date',
                    'optimization_label', 
                    'best_individual_min_period',
                    'best_individual_max_period',
                    'detrend_type', 
                    'best_fitness_value_sum', 
                    'best_individual_hp_filter_lambda', 
                    'best_individual_linear_filter_window_size_multiplier', 
                    'best_individual_final_kept_n_dominant_circles', 
                    'best_individual_num_samples'
                ]]


                result_list.append(result)


        final_result = pd.DataFrame()
        # Unisci i risultati in un unico DataFrame
        if result_list:  # Verifica che la lista non sia vuota
            final_result = pd.concat(result_list).sort_values(by=['detrend_type', 'best_individual_min_period'])

            # Rinomina le colonne
            final_result = final_result.rename(columns={
                'best_individual_min_period': 'min_period',
                'best_individual_max_period': 'max_period',
                'best_individual_hp_filter_lambda': 'hp_filter_lambda',
                'best_individual_final_kept_n_dominant_circles': 'final_kept_n_dominant_circles',
                'best_individual_num_samples': 'num_samples'
            })

            final_result.reset_index(inplace = True)


            if print_df_code:

                # genera struttura cicles_parameters
                result_string = "cicles_parameters = pd.DataFrame(columns = ['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda'])"

                for _, row in final_result.iterrows():
                    result_string += (
                        f"\ncicles_parameters.loc[len(cicles_parameters)] = ["
                        f"{row['num_samples']}, "
                        f"{row['final_kept_n_dominant_circles']}, "
                        f"{row.get('min_period', 'None')}, "
                        f"{row.get('max_period', 'None')}, "
                        f"{int(row['hp_filter_lambda'])}]"
                    )

                print(f"# Hyperparameers for {df['ticker_symbol'].unique()[0]} ticker:")
                print("# -----------------------------------------------------------")
                print(result_string)

        else:
            print("No results for past dates.")
            
        print(f"analysis_reference_date: {final_result.iloc[0]['analysis_reference_date']}")

        return final_result



    def detrend_lowess(self, signal, P_max, k=2):
        """
        Rimuove il trend da un segnale utilizzando LOWESS con una finestra configurabile.

        Parameters:
            signal (array): Il segnale da detrendere.
            time (array): L'array di tempo corrispondente al segnale.
            P_max (float): Il periodo massimo da conservare (i trend con periodi più lunghi saranno rimossi).
            k (float): Fattore moltiplicativo per determinare la finestra LOWESS. Default = 2.

        Returns:
            trend (array): Il trend stimato.
            residual (array): Il residuo (componente ciclica conservata).
        """
        # Determina la finestra basata sul periodo massimo e il fattore k
        time =  np.arange(len(signal))
        window = int(k * P_max)
        frac = window / len(time)  # Calcola la frazione del segnale per LOWESS

        # Calcola il trend usando LOWESS
        trend = lowess(signal, time, frac=frac, return_sorted=False)

        # Calcola il residuo (componente ciclica) - detrended signal
        residual = signal - trend

        return trend, residual




