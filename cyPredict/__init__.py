import yfinance as yf

# Basics
from enum import Enum
import sys
import traceback
from pathlib import Path

_NATIVE_MODULE_DIRS = [
    Path(__file__).resolve().parents[1] / "native" / "goertzel",
    Path(__file__).resolve().parents[1] / "native" / "cyfitness",
    Path(__file__).resolve().parents[1] / "native" / "cygaopt",
    Path(__file__).resolve().parents[1] / "native" / "cygaopt_multicore",
    Path(__file__).resolve().parents[1] / "native" / "genetic_optimization_legacy",
]

for _native_module_dir in _NATIVE_MODULE_DIRS:
    if _native_module_dir.exists():
        _native_module_path = str(_native_module_dir)
        if _native_module_path not in sys.path:
            sys.path.insert(0, _native_module_path)

from goertzel import goertzel_general_shortened as goertzel_general_shortened
from goertzel import goertzel_DFT as goertzel_DFT

from cyfitness import evaluate_fitness

import time
from decimal import Decimal
from scipy.signal.windows import tukey
from scipy.signal.windows import kaiser


# Math, sci and stats libraries
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
from scipy.integrate import simpson
from scipy.stats import pearsonr, spearmanr, kendalltau
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
import cyGAopt
from cyGAopt import run_genetic_algorithm
import cyGAoptMultiCore
from cyGAoptMultiCore import run_genetic_algorithm as run_genetic_algorithm_multicore

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
from plotly.offline import plot
from plotly.colors import hex_to_rgb
import plotly.io as pio
pio.renderers.default='iframe'

# Financial stats and indicators
import talib

# Stings manipulation
import re

# Filse manipulation
import os

# Processing capability
import multiprocessing

# Time Management

from datetime import datetime, timedelta, date
import pytz
from pytz import timezone
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay




class cyPredict:
    """Cycle-analysis engine for financial time series.

    The class downloads or loads OHLCV data, estimates dominant periods with
    Goertzel-based analysis, reconstructs projected cycle signals, and exposes
    helper workflows used by the research notebooks under
    ``D:\\Dropbox\\TRADING\\STUDIES DEVELOPMENT\\CYCLES ANALYSIS``.

    The public API is still legacy-compatible and intentionally broad. Many
    arguments are meaningful only for selected workflows; those mode-specific
    relationships are documented on the methods where they are consumed.
    """

    class Drive(Enum):
        local = 1
        GoogleDrive = 2
    
    class financialDataSource(Enum):
        csv_file = 1
        yfinance = 2

    def __init__(self,
                 data_source="yfinance",
                 data_filename = None,
                 ticker="SPY",
                 data_start_date="2004-01-01",
                 data_end_date=None,
                 data_timeframe="1d",
                 data_storage_path="\\cyPredict\\",
                 time_tracking = False,
                 output_clearing = False,
                 print_activity_remarks = True): 
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
        time_tracking : bool, default False
            Enables elapsed-time prints through ``track_time``.
        output_clearing : bool, default False
            Legacy notebook flag retained for workflows that clear notebook
            output between long processing steps.
        print_activity_remarks : bool, default True
            Enables verbose progress prints in selected workflows.

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
        ...     print_activity_remarks=False,
        ... )
        >>> cp.state["data_state"]
        'initialized'
        """

        # Instance variables (attributes)
        self.data_source = data_source
        self.data_filename = data_filename
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
            "data_state": 'not initialized',  # error, initialized
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
                
                # Se dati intraday → timezone definito dall'utente
                if 'm' in data_timeframe or 'h' in data_timeframe:
                    if self.data.index.tz is None:
                        self.data.index = self.data.index.tz_localize(self.original_data_time_zone) # ('UTC')  # L'utente deve aver configurato UTC
                # Se dati daily → rimuove timezone
                elif 'd' in data_timeframe or 'w' in data_timeframe or 'M' in data_timeframe:
                    self.data.index = self.data.index.tz_localize(None)

            elif data_source == "file":
                
                self.data = pd.read_csv(data_filename)
                
                # Converte la colonna Datetime in tipo datetime                
                self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
                
                # Imposta Datetime come indice e assicura ordinamento corretto
                self.data.set_index("Datetime", inplace=True)              
                self.data = self.data.sort_index()

                # Stampa prima ed ultima data
                self.data_start_date = self.data.index.min()
                self.data_end_date = self.data.index.max()
                print("First df Datetime:", self.data_start_date)
                print("Last df Datetime:", self.data_end_date)
                
            else:
                self.state["data_state"] = 'error'
                self.state["data_state_msg"] = 'Not managed data source ' + data_source
                print('Error: not managed data source ' + data_source)

            if(not self.data.empty):
                self.state["data_state"] = 'initialized'
                self.state["data_state_msg"] = 'Financial data is ready to be used.'
                
            else:
                self.state["data_state"] = 'error'
                self.state["data_state_msg"] = data_source + ' returned empty data, no error speficiation returned by the module. Look at standard output.'
        
            
        except Exception as e:  # ExceptionType should be Exception
            self.state["data_state"] = 'error'
            self.state["data_state_msg"] = str(e)
            print(f"An error occurred in download_finance_data: {e}")


    def hp_filter(self, data, lambda_, ret=False):
        nobs = len(data)

        output = output = data.to_numpy(dtype=np.float64).copy() #np.copy(data)

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
                print(f"[DEBUG] Z==0 at i={i}, skipping.")
                print(f"[DEBUG] i={i}, Z={Z}, a={a[i]}, b={b[i]}, c={c[i]}, H1={H1}, H2={H2}, H3={H3}")

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



    def find_next_valid_datetime(self, current_datetime, friday_times, saturday_times, sunday_times, workday_times, timezone=None):
        days = 0

        while True:
            temp_date = current_datetime + pd.DateOffset(days=days)

            if temp_date.weekday() == 4 and len(friday_times) > 0:
                day_times = friday_times
            elif temp_date.weekday() == 5 and len(saturday_times) > 0:
                day_times = saturday_times
            elif temp_date.weekday() == 6 and len(sunday_times) > 0:
                day_times = sunday_times
            elif len(workday_times) > 0:
                day_times = workday_times
            else:
                day_times = []

            if len(day_times) > 0:
                if days == 0:
                    # Giorno corrente: cerca orario successivo valido
                    current_time_index = np.searchsorted(day_times, current_datetime.time(), side='right')
                    if current_time_index < len(day_times):
                        next_time = day_times[current_time_index]
                        new_datetime = pd.Timestamp.combine(temp_date, next_time)
                        if timezone:
                            new_datetime = new_datetime.tz_localize(timezone)
                        return new_datetime
                    else:
                        # Nessun orario utile oggi: passa al giorno successivo
                        days += 1
                        continue
                else:
                    # Giorno successivo: usa il primo orario disponibile
                    next_time = day_times[0]
                    new_datetime = pd.Timestamp.combine(temp_date, next_time)
                    if timezone:
                        new_datetime = new_datetime.tz_localize(timezone)
                    return new_datetime

            days += 1



    def datetime_dateset_extend(self, df, extension_periods=10, timeframe  = None):

        timezone = df.index.tz
        today = df.index.max()  # ✅ Mantiene anche l'orario

        if timeframe is None:
            # Stima grezza del timeframe medio (in secondi)
            deltas = df.index.to_series().diff().dropna()
            median_delta = deltas.median()

            if median_delta >= pd.Timedelta(days=1):
                timeframe = '1d'
            else:
                timeframe = 'intraday'  # Assume tutto < 1d come intraday
                
                
            if timeframe in ['1d', '1h', '1wk', '1mo']:
                print("DAILY TIMEFRAME DETECTED")

                # Determina quali giorni della settimana sono presenti storicamente
                historical_weekdays = set(df.index.weekday)

                last_date = df.index.max().normalize()
                new_indexes = []

                next_date = last_date + pd.Timedelta(days=1)

                # Continua finché hai aggiunto tutte le extension_periods necessarie
                while len(new_indexes) < extension_periods:
                    # Se il giorno della settimana della prossima data è presente nello storico, aggiungila
                    if next_date.weekday() in historical_weekdays:
                        new_indexes.append(next_date)

                    # Passa al giorno successivo
                    next_date += pd.Timedelta(days=1)

                # Gestione timezone se necessario
                if timezone is not None:
                    new_indexes = [pd.Timestamp(d).tz_localize(timezone) for d in new_indexes]

                
            else:
                # Prendi TUTTI i datetime dell'ultimo venerdì disponibile
                print("INTRADAY TIMEFRAME DETECTED")

                today = df.index.max().date()

                # Prendi TUTTI i datetime dell'ultimo venerdì disponibile (prima di oggi)
                last_friday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 4)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_friday):
                    friday_times = df.loc[df.index.date == last_friday].index.time
                else:
                    friday_times = []

                # Prendi TUTTI i datetime dell'ultimo sabato disponibile (prima di oggi)
                last_saturday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 5)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_saturday):
                    saturday_times = df.loc[df.index.date == last_saturday].index.time
                else:
                    saturday_times = []

                # Prendi TUTTI i datetime dell'ultima domenica disponibile (prima di oggi)
                last_sunday = (
                    df.loc[(df.index.date < today) & (df.index.weekday == 6)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_sunday):
                    sunday_times = df.loc[df.index.date == last_sunday].index.time
                else:
                    sunday_times = []

                # Prendi TUTTI i datetime dell'ultimo giorno lavorativo disponibile (prima di oggi, esclusi venerdì e domenica)
                last_workday = (
                    df.loc[(df.index.date < today) & (df.index.weekday != 4) & (df.index.weekday != 6)]
                    .index.to_series()
                    .dt.date
                    .max()
                )
                if pd.notna(last_workday):
                    workday_times = df.loc[df.index.date == last_workday].index.time
                else:
                    workday_times = []
                    
                    
                samples_per_friday = len(friday_times)
                samples_per_sunday = len(sunday_times)
                samples_per_workday = len(workday_times)

                # Ricava l’ultimo timestamp reale
                last_real_timestamp = df.index.max()

                # Ricava tutti i datetime dei giorni settimanali campione
                new_indexes = []
                new_datetime = self.find_next_valid_datetime(
                    last_real_timestamp, friday_times, saturday_times, sunday_times, workday_times, timezone
                )

                for _ in range(extension_periods):
                    new_indexes.append(new_datetime)
                    new_datetime = self.find_next_valid_datetime(
                        new_datetime, friday_times, saturday_times, sunday_times, workday_times, timezone
                    )
                    

        len_before = len(df)

        # Crea solo nuove righe (non modifica df esistente)
        new_rows = pd.DataFrame(np.nan, index=new_indexes, columns=df.columns)
        df = pd.concat([df, new_rows])
        
        len_after = len(df)
        
        df.to_csv('C:\\Users\\Federico\\Downloads\\df new dates.csv')

        return df



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
                             time_tracking = False,
                             print_activity_remarks = False 
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
        time_tracking : bool, default False
            Enables elapsed-time prints.
        print_activity_remarks : bool, default False
            Enables verbose progress prints and displayed intermediate rows.

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

        
        print(f'Data column name: {data_column_name}')
        print(f'windowing {windowing}, kaiser_beta {kaiser_beta}')
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
        self.print_activity_remarks = print_activity_remarks


        elaborated_data_series = [] # pd.DataFrame()
        signals_results_series = [] # pd.DataFrame()
        configurations_series = []
        goertzel_amplitudes = []
        
        if(detrend_type != 'linear' and detrend_type != 'lowess'):
            detrend_type = 'hp_filter'


        if(self.print_activity_remarks):  
            print("----------------------------------------------------")
            print("   Starting multiperiod_analysis, parameters:")
            print("----------------------------------------------------")
    
            for k, v in locals().items():
                print(f"{k} = {v}")

            print('\nMultiperiod analysis')            
            self.track_time('1. multiperiod_analysis: entering for loop, start calling analyze_and_plot')
            

        # Optionally reduce the dataframe before single-range detrending.
        original_data = self.data  # non modificato
            
        if cut_to_date_before_detrending:
            max_range_row = periods_pars.loc[periods_pars['max_period'].idxmax()]
            protective_length = int(1.5 * max_range_row['num_samples'] )
        
            valid_times = original_data.index[original_data.index <= current_date]
            if len(valid_times) == 0:
                raise ValueError("⚠️ Nessun dato disponibile prima di current_date nel dataset.")
            
            last_valid_time = valid_times.max()
            idx_max_time = original_data.index.get_indexer([last_valid_time])[0]
            start_idx = max(0, idx_max_time - protective_length)
        
            reduced_data = original_data.iloc[start_idx:]
            print(f"🔒 Protective cut: using last {len(reduced_data)} samples from index {start_idx} to {idx_max_time}")
            print(f"\tStart date     : {reduced_data.index[0]}")
            print(f"\tEnd date       : {reduced_data.index[-1]}")
            print(f"\tCurrent date   : {current_date}")
        else:
            print(f"original dataset has not been reduced: using {len(reduced_data)} samples from index {start_idx} to {idx_max_time}")
            reduced_data = original_data

            
        
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
                
             
            print(f'\nStarted periods analysis in range[{min_period}, {max_period}]')
            if(self.print_activity_remarks):
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
                             time_tracking = time_tracking
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
        if(self.print_activity_remarks):
            print('\nRe-Factorization of Cyrcles Amplitude')        
            self.track_time('\n2. multiperiod_analysis: starting Re-Factorize Cyrcles Amplitude')

        # Select the reference detrended series for optimizer fitness.
        if(reference_detrended_data == "less_detrended"):
            if(detrend_type == 'hp_filter'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['hp_filter_lambda'])
            if(detrend_type == 'lowess'):
                index_detrended_data = max(range(len(configurations_series)), key=lambda i: (configurations_series[i]['lowess_k'] * configurations_series[i]['max_period']))

        # Alternative reference: use the longest single-range analysis.
        if(reference_detrended_data == "longest"):
            index_detrended_data = max(range(len(configurations_series)), key=lambda i: configurations_series[i]['num_samples'])
            
        print(f'index_detrended_data {index_detrended_data}')

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
        
        if(self.print_activity_remarks):
            self.track_time('3. multiperiod_analysis: end Re-Factorize Cyrcles Amplitude')
        
        

        # -------------------------------------------------------
        # Re-estimate amplitudes with Goertzel on the reference signal
        # -------------------------------------------------------
        
        if(self.print_activity_remarks):
            self.track_time('\n4. multiperiod_analysis: starting Goertzel best amplitudes')

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
        
        if(self.print_activity_remarks):
            self.track_time('\n5. multiperiod_analysis: end Goertzel best amplitudes')
        
        
        
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
                
                comparision_length = int(2.5 * row['peak_periods']) # comparision length
                length = (len_series - row['start_rebuilt_signal_index'])
                start_comparison_index = len_series - comparision_length
                
                # Sweep candidate amplitudes in descending order.
                for temp_amp in np.arange(self.MultiAn_detrended_max, 0, -0.01):                    
                    
                    
                    temp_circle_signal = pd.Series([0.0] * len_series)
                    time = np.linspace(0, length, length, endpoint=False)

                    temp_circle_signal[row['start_rebuilt_signal_index']:] = temp_amp * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases'])

                    temp_rebuilt_signal = composite_dominant_cycle_signal + temp_circle_signal
                    
                    error = mean_squared_error(self.MultiAn_reference_detrended_data[start_comparison_index:], temp_rebuilt_signal[start_comparison_index:]) 
                    
                   
                    if(error < last_error):                        
                        best_amplitude = temp_amp 
                        best_error = error

                    last_error = error
                    
                    
                # Store the best amplitude found for this cycle.
                amplitudes.append(best_amplitude)
                best_fitness_value = best_error
                print('Period ' + str(row['peak_periods']) + f', best amplitude {best_amplitude}, best_fitness_value {best_fitness_value}')
                
                # Add this cycle to the cumulative rebuilt signal.
                composite_dominant_cycle_signal = composite_dominant_cycle_signal + temp_rebuilt_signal
           
            if(self.print_activity_remarks):    
                print('Single cycle best fitting, aplitudes:')
                print(amplitudes)
            
        # -------------------------------------------------------
        # DEAP optimizer for amplitudes and optional frequency/phase tuning
        # -------------------------------------------------------
        
        elif(opt_algo_type == 'genetic_omny_frequencies'):

            
            self.debug_check_complex_values()

            
 
            self.track_time('\n6. Genetics algo for best amplitudes identification, start inizializing')
    

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
                print(f"CPU count: {cpu_count}")

                pool = multiprocessing.Pool()
                toolbox.register("map", pool.map)


            # Create the initial population.
            population = toolbox.population(n=population_n)
            
            self.track_time('6. Genetics algo for best amplitudes identification, end inizialization')
        

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


            if(self.print_activity_remarks):
                print("\n\n--------------------------------------------------------")
                print("Multirange Analysis Genetics Optimization results:")
                print('\tbest_individual: ' + str(best_individual))
                print('\tbest_fitness: ' + str(best_fitness))
                print("--------------------------------------------------------")
                


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

            display(self.MultiAn_dominant_cycles_df)



        # -----------------------------------------------------------------------------------
        # C++ genetic optimizer for amplitudes and optional frequency/phase tuning
        # -----------------------------------------------------------------------------------

        elif opt_algo_type == 'cpp_genetic_amp_freq_phase':

            print(f"self.MultiAn_dominant_cycles_df columns: {self.MultiAn_dominant_cycles_df.columns}")
            
            self.track_time('\n6. C++ Genetic Optimization start')
        
            n_cycles = len(self.MultiAn_dominant_cycles_df)
            
            if 'single_range_goertzel_peak_amplitudes' in self.MultiAn_dominant_cycles_df.columns:
                scaler = MinMaxScaler(feature_range=(0, detrended_abs_max))
                amp_init = scaler.fit_transform(self.MultiAn_dominant_cycles_df['single_range_goertzel_peak_amplitudes'].to_numpy().reshape(-1, 1)).flatten()

            else:
                amp_init = np.array([0.1] * n_cycles)

            freq_init = self.MultiAn_dominant_cycles_df['peak_frequencies'].to_numpy()
            
            phase_init = self.MultiAn_dominant_cycles_df['peak_phases'].to_numpy()

            print(f"amp_init: {amp_init}")


            initial_vector = []
            
            # Amplitudes.
            if amplitudes_inizialization_type == "random":
                initial_random_amplitudes = True
                initial_vector += amp_init.tolist()  # placeholder, saranno ignorate nel cpp
                
            elif amplitudes_inizialization_type == "all_equal_middle_value":
                initial_random_amplitudes = False
                middle_value = (self.MultiAn_detrended_max - self.MultiAn_detrended_min) / 2
                initial_vector += [middle_value] * n_cycles

                print(f"Amplitudes middle vale {middle_value}")
                
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

            print(f"detrended_abs_max {detrended_abs_max}")
        
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

        
            lb = []
            ub = []

            lb = amp_min.copy()
            ub = amp_max.copy()
            
            if self.frequencies_ft:
                lb += freq_min
                ub += freq_max
            
            if self.phases_ft:
                lb += phase_min
                ub += phase_max

        
            
            def fitness_func_cpp(flat_list):
                fitness_result = self.MultiAn_evaluateFitness(flat_list, False)
                return float(fitness_result[0]) if isinstance(fitness_result, tuple) else float(fitness_result)


            if(enabled_multiprocessing):

                print('run_genetic_algorithm_multicore')


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
                    period_multiplier=self.period_related_rebuild_multiplier
                )




            else:

                print('run_genetic_algorithm')
        
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
                    n_cycles 
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




            if self.frequencies_ft and self.phases_ft:
                individual = np.concatenate([amp, freq, phase])
            elif self.frequencies_ft:
                individual = np.concatenate([amp, freq])
            elif self.phases_ft:
                individual = np.concatenate([amp, phase])
            else:
                individual = amp

            
            self.MultiAn_dominant_cycles_df['best_fitness'] = self.MultiAn_evaluateFitness(individual, False)


        
            if self.print_activity_remarks:
                print("\n\n--------------------------------------------------------")
                print("C++ Genetic Optimization results:")
                print('\tbest_fitness: ' + str(self.MultiAn_dominant_cycles_df["best_fitness"].iloc[0]))
                print("--------------------------------------------------------")
        
            self.track_time('6. C++ Genetic Optimization end')
        

            best_fitness_value = self.MultiAn_dominant_cycles_df['best_fitness']


        # -----------------------------------------------------------------------------------
        # NLopt optimizer for amplitudes and optional frequency/phase tuning
        # -----------------------------------------------------------------------------------

        elif(opt_algo_type == 'nlopt_amplitudes_freqs_phases'):
            
            self.track_time('\n6. NLopt optimization: start')
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


            


            print(f'Best fitness: {best_fitness_value}')


            self.track_time('6. NLopt optimization: end')


        # -------------------------------------------------------------------------------------------------------------------------
        # Hyperopt TPE/ATPE optimizer for amplitudes and optional frequency/phase tuning
        # -------------------------------------------------------------------------------------------------------------------------

        elif(opt_algo_type ==  'tpe' or opt_algo_type ==  'atpe'):

            from hyperopt import STATUS_OK  # può anche stare fuori dalla funzione una volta sola

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

            print("Best Amplitudes:", best_amplitudes)
            
            if self.frequencies_ft:
                frequencies = [best[f'frequency_{i}'] for i in range(n)]
                self.MultiAn_dominant_cycles_df['frequency'] = frequencies
                self.MultiAn_dominant_cycles_df['best_frequencies'] = frequencies
                print("Best frequencies:", frequencies)
            
            if self.phases_ft:
                phases = [best[f'phase_{i}'] for i in range(n)]
                self.MultiAn_dominant_cycles_df['phase'] = phases
                self.MultiAn_dominant_cycles_df['best_phases'] = phases
                print("Best phases:", phases)



            # Re-evaluate the best point and store the resulting fitness.
            best_fitness_value = objective(best)
            self.MultiAn_dominant_cycles_df['best_fitness'] = best_fitness_value

            display(self.MultiAn_dominant_cycles_df)

            amplitudes = best_amplitudes

        else:

            print("Errore: optmization type not in list ('mono_frequency', 'genetic_omny_frequencies', 'tpe', 'atpe')")
            return None, None, None, None, None, None


        # -----------------------------------------------------------------------
        # Create composite signals from optimized cycle parameters
        # -----------------------------------------------------------------------

        if(self.print_activity_remarks):
            self.track_time('\n9. Genetics start composite signal creation')


        temp_circle_signal = []
        composite_dominant_cycle_signal = []

        # Projection length follows the longest generated single-range output.
        len_series = len(self.data) #len(self.MultiAn_reference_detrended_data)

        # Use the longest single-range index as the composite signal index.
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

        

        start_index = index_of_max_time_for_cd - num_samples    


        # -------------------------------------------------
        # Scale composite and alignment signals for output
        # -------------------------------------------------
        
        if(self.print_activity_remarks):
            print("\nSignals scaling")
            self.track_time('\n11. Genetics end composite signal creation')

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

        if(self.print_activity_remarks):
            self.track_time('12. Genetics end composite signal creation')
            
   

        print(f"BEFORE CHART PLOTTING, index_of_max_time_for_cd {index_of_max_time_for_cd}")
        # -------------------------------------------------
        # Optional Plotly diagnostics
        # -------------------------------------------------

        if(self.print_activity_remarks):
            print('Before CHARTS PLOT')            
            

        if(show_charts == True):

            # Original data, detrended, dominant circles signal, averages delta
            
            
            fig = make_subplots(rows=3,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.03,
                                subplot_titles=("Original Data - " + 
                                                self.ticker + " " + 
                                                data_column_name + 
                                                " Price",
                                                "Composite Domaninant Cycles Signal",
                                                "Cycles Alignment Indicators",
                                ))
            


            fig.add_trace(go.Scatter(x=reduced_data .index,
                                     y=reduced_data [data_column_name],
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
 
            
            fig.add_vline(
                x=index_of_max_time_for_cd,
                line=dict(color="red", dash="dot"),
                name="Current Date",
                row=1, col=1
            )
            
            fig.add_vline(
                x=index_of_max_time_for_cd,
                line=dict(color="red", dash="dot"),
                name="Current Date",
                row=2, col=1
            )
            
            fig.add_vline(
                x=index_of_max_time_for_cd,
                line=dict(color="red", dash="dot"),
                name="Current Date",
                row=3, col=1
            )
            
            # Zoom iniziale centrato sulla current time
            samples_visible_before = 80
            samples_visible_after = 80

            index_center = index_of_max_time_for_cd
            start_range = max(0, index_center - samples_visible_before)
            end_range = index_center + samples_visible_after

            fig.update_xaxes(range=[start_range, end_range])
            fig.update_xaxes(type="category")

            
            fig.update_layout(
                title="Goertzel Dominant Cyrcles Analysis",
                height=1000,
                autosize=True,
                margin=dict(l=40, r=40, t=240, b=40),  
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.2,  
                    xanchor="center",
                    x=0.5
                )
            )
            
            fig.update_layout(
                yaxis=dict(
                    autorange=True,
                    fixedrange=False
                )
            )

            # Trova massimi e minimi locali
            y = scaled_composite_signal
            x = elaborated_data_series[max_length_series_index].index

            max_indices = argrelextrema(y, np.greater)[0]
            min_indices = argrelextrema(y, np.less)[0]
            
            # Converte gli indici in datetime
            max_datetimes = x[max_indices]
            min_datetimes = x[min_indices]
            
            # Filtra i 2 massimi/minimi successivi e 1 precedente
            def pick_extrema_near_target(datetimes, target_index, count_after=2, count_before=1):
                target_dt = x[target_index]
                dt_series = pd.Series(datetimes)
                after = dt_series[dt_series > target_dt].sort_values().head(count_after)
                before = dt_series[dt_series <= target_dt].sort_values(ascending=False).head(count_before)
                return before.tolist() + after.tolist()
            
            relevant_max_dt = pick_extrema_near_target(max_datetimes, index_of_max_time_for_cd)
            relevant_min_dt = pick_extrema_near_target(min_datetimes, index_of_max_time_for_cd)
            
            # Annotazioni
            used_points = []
            for dt in relevant_max_dt:
                y_val = scaled_composite_signal[x.get_loc(dt)]
                count_same_y = sum(1 for xd, yd in used_points if abs(yd - y_val) < 0.01 and abs((dt - xd).total_seconds()) < 60 * 60)
                offset_y = 0.02 + count_same_y * 0.05     # 0.07, 0.12, ...
                ay_val = -30 - count_same_y * 15          # visivamente separati

                used_points.append((dt, y_val))
            
                fig.add_annotation(
                    x=dt,
                    y=y_val + offset_y,
                    text=dt.strftime('%H:%M') if (pd.Series(reduced_data.index).diff().mode()[0] < pd.Timedelta("1D")) else dt.strftime('%Y-%m-%d'),
                    showarrow=True,
                    arrowhead=2,
                    arrowside="end",
                    arrowcolor="red",
                    arrowwidth=2.5,
                    ax=0,
                    ay=ay_val,
                    row=2,
                    col=1
                )            
                        
            # Annotazioni sui minimi (alternanza: vicino–lontano–vicino)
            used_points = []
            for dt in relevant_min_dt:
                y_val = scaled_composite_signal[x.get_loc(dt)]
                count_same_y = sum(1 for xd, yd in used_points if abs(yd - y_val) < 0.01 and abs((dt - xd).total_seconds()) < 60 * 60)
                offset_y = 0.02 + count_same_y * 0.05     # 0.07, 0.12, ...
                ay_val = 30 + count_same_y * 15          # visivamente separati
                used_points.append((dt, y_val))
            
                fig.add_annotation(
                    x=dt,
                    y=y_val - offset_y,
                    text=dt.strftime('%H:%M') if (pd.Series(reduced_data.index).diff().mode()[0] < pd.Timedelta("1D")) else dt.strftime('%Y-%m-%d'),
                    showarrow=True,
                    arrowhead=2,
                    arrowside="end",
                    arrowcolor="green",
                    arrowwidth=2.5,
                    ax=0,
                    ay=ay_val,
                    row=2,
                    col=1
                )



            # Ottieni datetime visibili
            # Datetime centrato
            x_range_start = reduced_data .index[start_range]
            x_range_end = reduced_data .index[min(end_range, len(reduced_data .index)-1)]
            
            # Trova i veri limiti di slicing con ricerca binaria
            idx_start = reduced_data .index.searchsorted(x_range_start)
            idx_end = reduced_data .index.searchsorted(x_range_end)
            
            visible_y = reduced_data .iloc[idx_start:idx_end][data_column_name].dropna()

              
            if not visible_y.empty:
                ymin = visible_y.min()
                ymax = visible_y.max()
                range_y = ymax - ymin
            
            

                fig.update_yaxes(
                    range=[ymin - 10, ymax + 10],
                    autorange=False,
                    fixedrange=False,
                    row=1,
                    col=1
                )
            
            # Ottieni i datetime visibili già calcolati sopra
            visible_x_range = x[idx_start:idx_end]
            visible_cdc = pd.Series(scaled_composite_signal, index=x).loc[visible_x_range].dropna()
            
            if not visible_cdc.empty:
                ymin2 = visible_cdc.min()
                ymax2 = visible_cdc.max()
                range_y2 = ymax2 - ymin2
            
                fig.update_yaxes(
                    range=[ymin2 - 20, ymax2 + 20],
                    autorange=False,
                    fixedrange=False,
                    row=2,
                    col=1
                )

            

            # Unione ordinata dei picchi etichettati
            all_extremes = sorted(relevant_min_dt + relevant_max_dt)
            
            # Crea bande colorate tra coppie alternate
            for i in range(len(all_extremes) - 1):
                t0 = all_extremes[i]
                t1 = all_extremes[i + 1]
                y0, y1 = 0, 1  # domain intero
                
                # Colori corretti: rosso dopo massimo, verde dopo minimo
                color = "rgba(255, 0, 0, 0.2)" if t0 in relevant_max_dt else "rgba(0, 255, 0, 0.2)"
                
                for r in [1, 2, 3]:
                    fig.add_vrect(
                        x0=t0, x1=t1,
                        fillcolor=color,
                        opacity=0.2,
                        line_width=0,
                        row=r,
                        col=1
                    )
            
            # Intorno grigio ±3 o ±4
            x_list = list(x)
            all_tagged_extremes = sorted(relevant_min_dt + relevant_max_dt)
            past_extremes = [dt for dt in all_tagged_extremes if dt <= x[index_of_max_time_for_cd]]
            future_extremes = [dt for dt in all_tagged_extremes if dt > x[index_of_max_time_for_cd]]
            
            for i, dt in enumerate(past_extremes + future_extremes):
                center_idx = x_list.index(dt)
                if dt in past_extremes or i == len(past_extremes):  # primo futuro
                    delta = 3
                else:
                    delta = 4
            
                start_idx = max(0, center_idx - delta)
                end_idx = min(len(x_list) - 1, center_idx + delta)
                t0 = x_list[start_idx]
                t1 = x_list[end_idx]
            
                for r in [1, 2, 3]:
                    fig.add_vrect(
                        x0=t0, x1=t1,
                        fillcolor="rgba(150,150,150,0.60)",
                        opacity=0.25,
                        line_width=0,
                        row=r,
                        col=1
                    )


            # Visualizza il secondo grafico con i subplot
            fig.show()            
            
            plot(fig, filename=f'multirange analysis for {self.ticker}.html', auto_open=False)
            
            print(f"Type of reduced_data .index: {type(reduced_data .index)}")
            print(f"Timezone of reduced_data .index: {reduced_data .index.tz}")

            print(f"Type of elaborated_data_series index: {type(elaborated_data_series[max_length_series_index].index)}")
            print(f"Timezone of elaborated_data_series index: {elaborated_data_series[max_length_series_index].index.tz}")




        return elaborated_data_series, signals_results_series, composite_signal, configurations_series, None, None, index_of_max_time_for_cd, scaled_signals, best_fitness_value


    def debug_check_complex_col(self, colname):
        """ Stampa le righe con valori complessi nella colonna colname. """
        if colname not in self.MultiAn_dominant_cycles_df.columns:
            return
    
        df = self.MultiAn_dominant_cycles_df
        complex_mask = df[colname].apply(lambda val: isinstance(val, complex))
        if complex_mask.any():
            indices = df.index[complex_mask]
            for i in indices:
                val = df.at[i, colname]
                print(f"  -> index={i}, {colname}={val}")


    def debug_check_complex_values(self):
        """ Richiama debug_check_complex_col su ampiezze/frequenze/fasi (se esistono). """
        cols = ["peak_amplitudes", "peak_frequencies", "peak_phases"]
        for c in cols:
            self.debug_check_complex_col(c)


    def get_goertzel_amplitudes(self):

        return self.goertzel_amplitudes


    def cicles_composite_signals(self, max_length_series, amplitudes, MultiAn_dominant_cycles_df, df_indexes_list, composite_signal_column_name):
        

        composite_signal = pd.DataFrame(index=df_indexes_list)
        composite_signal[composite_signal_column_name] = 0.0

        max_start_rebuilt_signal_index = 0

        for index, row in MultiAn_dominant_cycles_df.iterrows():

            temp_period = row['peak_periods'] #  1/row['peak_frequencies']
            new_column_name = composite_signal_column_name + '_refact_dominant_circle_signal_period_' + str(temp_period)
            composite_signal[new_column_name] = 0.0

            remanant_length = np.int64(max_length_series - row['start_rebuilt_signal_index'])
            time = np.linspace(0, remanant_length, remanant_length, endpoint=False)

            # Composite Cycles from GA amplitudes refactoring
            composite_signal.iloc[int(row['start_rebuilt_signal_index']):, composite_signal.columns.get_loc(new_column_name)] = amplitudes[index] * np.sin(2 * np.pi * row['peak_frequencies'] * time + row['peak_phases']) 
            composite_signal[composite_signal_column_name]  += composite_signal[new_column_name]

            if(max_start_rebuilt_signal_index < int(row['start_rebuilt_signal_index'])):
                max_start_rebuilt_signal_index = int(row['start_rebuilt_signal_index'])
 
        return composite_signal



    def indict_MACD_SGMACD(self, data, data_column_name, dominant_period, macd_slow_ratio = 26, macd_signal_ratio = 9):

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


    def indict_RSI_SG_smooth_RSI(self, data, data_column_name, end_rebuilt_signal_index, dominant_period = 10):

        data_len = len(data)
        indicators_period = int(dominant_period / 2)
        data['RSI_' + str(indicators_period)] = talib.RSI(data[data_column_name], indicators_period)
        data['RSI_' + str(indicators_period)] = data['RSI_' + str(indicators_period)] .fillna(0)

        polyorder = 2
        if(indicators_period < 3):
            polyorder = 1

        data['smoothed_RSI_' + str(indicators_period)] = pd.Series([np.nan] * data_len)
        data.iloc[0:end_rebuilt_signal_index+1, data.columns.get_loc('smoothed_RSI_' + str(indicators_period))] = savgol_filter(data['RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1], indicators_period, polyorder)  #   ['smoothed_RSI_' + str(indicators_period)].iloc[0:end_rebuilt_signal_index+1] = savgol_filter(data['RSI_' + str(indicators_period)][0:end_rebuilt_signal_index+1], indicators_period, polyorder)

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
    
    def custom_crossover(self, ind1, ind2):
        
        if len(ind1) > 1 and len(ind2) > 1:
            # Esegui il crossover a due punti
            return tools.cxTwoPoint(ind1, ind2)
        else:
            # Gestisci caso di individui con un solo gene
            # Ad esempio, non fare nulla o applica una mutazione
            return tools.cxUniform(ind1, ind2, 0.5)



    def indict_centered_average_deltas(self, data, data_column_name, dominant_period = 10):

        long_average = data[data_column_name].rolling(window=round(dominant_period), center = True).mean()
        long_average.fillna(0, inplace=True)
        short_average = data[data_column_name].rolling(window=round(dominant_period/2), center = True).mean()
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
        len_before = len(signal)



        rebuilt_sig_left_zeros = np.zeros(start_rebuilt_signal_index ) #+ 1
        signal = np.concatenate((rebuilt_sig_left_zeros, signal), axis = 0)

        
        
        len_after_left = len(signal)
        


        rebuilt_sig_right_zeros = 0

        # Add zeros on the right side if signal end before the last original_data element
        if(total_length  > len_after_left):

            rebuilt_sig_right_zeros = np.zeros(total_length - len_after_left)
            signal = np.concatenate((signal, rebuilt_sig_right_zeros), axis = 0)
            # this ensures to limit the added rebuilt signal to the max current time (last sample index of original data)
            signal = signal[0:(total_length)]
        

        
        projection_periods_extetions = 0
        if(len(signal) > total_length):
            projection_periods_extetions = len(signal) - total_length
            

            
        # Se i segnaposti a destra non bastano
        projection_periods_extensions = 0
        if len(signal) > total_length:
            projection_periods_extensions = len(signal) - total_length
        

        return signal, projection_periods_extetions


    def get_row_score(self, row):

        scores = row.sum()

        return scores



    def get_gloabl_score(self, data, ascending_columns, descending_columns):

        df = pd.DataFrame(data)
        data_ascending = pd.DataFrame()
        data_descending = pd.DataFrame()
        global_score = pd.DataFrame()

        # Seleziona solo le colonne desiderate da data

        data_ascending = df[ascending_columns].rank(ascending=True, axis=0)
        data_descending = df[descending_columns].rank(ascending=False, axis=0)

        global_score = pd.concat([data_ascending, data_descending], axis=1)
        global_score['global_score'] = global_score.apply(self.get_row_score, axis=1)

        return global_score['global_score']



    def trade_predicted_dominant_cicles_peaks(self,
                                               current_date,
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
                                                other_correlations = True,
                                                show_charts = False,
                                                print_report = False,
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
                                     num_samples,
                                     final_kept_n_dominant_circles,
                                     min_period,
                                     max_period,
                                     hp_filter_lambda,
                                     opt_algo_type = 'genetic_omny_frequencies', # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency', 'tpe', 'atpe'
                                     detrend_type = 'hp_filter', #'linear', #'hp_filter',
                                     windowing = None,
                                     kaiser_beta = 3,
                                     linear_filter_window_size_multiplier = 0.7,
                                     period_related_rebuild_range = True,
                                     period_related_rebuild_multiplier = 2.5
                                     ):
        


        cycles_parameters = pd.DataFrame(columns = ['num_samples', 'final_kept_n_dominant_circles', 'min_period', 'max_period', 'hp_filter_lambda'])
        cycles_parameters.loc[len(cycles_parameters)] = [num_samples, final_kept_n_dominant_circles, min_period, max_period, hp_filter_lambda]
    
        
        
        (elaborated_data_df, 
         signals_results_df, 
         composite_signal, 
         configurations, 
         bb_delta, cdc_rsi, 
          index_of_max_time_for_cd,
          scaled_signals,
          best_fitness_value) = self.multiperiod_analysis(
                                                        data_column_name = 'Close',
                                                        current_date = current_date,
                                                        periods_pars = cycles_parameters,
                                                        population_n = 10,
                                                        CXPB = 0.7,
                                                        MUTPB = 0.3,
                                                        NGEN = 30,
                                                        MultiAn_fitness_type = "mse",
                                                        MultiAn_fitness_type_svg_smoothed = False,
                                                        MultiAn_fitness_type_svg_filter = 4,
                                                        reference_detrended_data = "less_detrended", # longest less_detrended
                                                        enable_cycles_alignment_analysis = False,
                                                        opt_algo_type = opt_algo_type, #'genetic_omny_frequencies', # 'genetic_omny_frequencies', 'genetic_frequencies_ranges','mono_frequency', 'tpe', 'atpe'
                                                        detrend_type = detrend_type, #'linear', #'linear', #'hp_filter',
                                                        windowing = windowing,
                                                        kaiser_beta = kaiser_beta,
                                                        linear_filter_window_size_multiplier = linear_filter_window_size_multiplier, #0.7,
                                                        period_related_rebuild_range = period_related_rebuild_range, #True,
                                                        period_related_rebuild_multiplier = period_related_rebuild_multiplier, #2.5,
                                                        show_charts = False,
                                                        enabled_multiprocessing = False, # double level multiprocessing is not allowed 
                                                        time_tracking = self.time_tracking
                                                        )



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
                                          windowing = None,
                                          kaiser_beta = 3,                                             
                                          linear_filter_window_size_multiplier = 0.7,
                                          period_related_rebuild_range = True,
                                          period_related_rebuild_multiplier = 2.5):

        last_date_index = data.index.get_loc(last_date)

        fitness = 0
        count = 0

        for index in range(periods_number):
            rel_pos  = last_date_index - periods_number + index
            current_date = data.index[rel_pos]

            print(f'in CDC_vs_detrended_correlation_sum, last_date {last_date}')
            print(f'in CDC_vs_detrended_correlation_sum, current_date {current_date}')
            print(f'Day n.//datetime period {index}, calling CDC_vs_detrended_correlation()')
            print(f'\t periods_number {periods_number}, num_samples {num_samples}, final_kept_n_dominant_circles {final_kept_n_dominant_circles}, hp_filter_lambda {hp_filter_lambda}, period_related_rebuild_range {period_related_rebuild_range},  period_related_rebuild_multiplier {period_related_rebuild_multiplier}, linear_filter_window_size_multiplier {linear_filter_window_size_multiplier}')
            
            start = self.MultiAn_dominant_cycles_df['start_rebuilt_signal_index'].min()
            end = self.MultiAn_dominant_cycles_df['end_rebuilt_signal_index'].max()

            rebuilt_segment = rebuilt_signal[start:end]
            reference_segment = self.detrended_data[start:end]

            print(f"Checking rebuilt_signal[{start}:{end}] NaNs: {rebuilt_segment.isna().sum()}")
            print(f"Checking detrended_data[{start}:{end}] NaNs: {reference_segment.isna().sum()}")

            if rebuilt_segment.isna().any() or reference_segment.isna().any():
                print("⚠️ Detected NaNs in one of the compared series during fitness evaluation.")


            temp_fitness = self.CDC_vs_detrended_correlation(
                                                            current_date = current_date,
                                                            num_samples= num_samples,
                                                            final_kept_n_dominant_circles=final_kept_n_dominant_circles,
                                                            min_period = min_period,
                                                            max_period = max_period,
                                                            hp_filter_lambda = hp_filter_lambda,   
                                                            windowing = windowing,
                                                            kaiser_beta = kaiser_beta,   
                                                            opt_algo_type = opt_algo_type, 
                                                            detrend_type = detrend_type,
                                                            linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                            period_related_rebuild_range = period_related_rebuild_range,
                                                            period_related_rebuild_multiplier = period_related_rebuild_multiplier
                                                           )
            

            fitness += temp_fitness

            if(temp_fitness > 0):
                count += 1
                

        if(count > 0):
            fitness = fitness / count


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
                print(f"Aggiusto x0[{i}]={x0[i]} che tocca/supera ub={ub[i]}")
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
            print(f"Using NLopt algorithm: COBYLA")
        else:
            algo = nlopt.GN_ISRES
            print(f"Using NLopt algorithm: ISRES (fallback for high dimension)")
        
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
                    print('period_related_rebuild_index < start_rebuilt_signal_index')
                    
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



    def MultiAn_cyclesAlignKPI(self, signals, start_position, weights = None, periods = None):

        last_position = len(signals) #- 1

        # Pre-fill positions before the first rebuildable index with zeros.
        kpi_series = pd.Series([0] * start_position, dtype=np.int64)
        weigthed_kpi_series = pd.Series([0] * start_position, dtype=np.int64)
        
        
        peaks_min_df = {} #pd.DataFrame()
        peaks_max_df = {} # pd.DataFrame()
        
        # Cache extrema indexes for each component signal once.
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

                # Reuse precomputed local minima and maxima.

                peaks_min = peaks_min_df[column]
                peaks_max = peaks_max_df[column]
                

                # Candidate extrema before the current position.
                peaks_min_before = [peak for peak in peaks_min if peak < position]
                peaks_max_before = [peak for peak in peaks_max if peak < position]

                # Candidate extrema after the current position.
                peaks_min_after = [peak for peak in peaks_min if peak > position]
                peaks_max_after = [peak for peak in peaks_max if peak > position]

                # Nearest previous minimum.
                previous_peak_index_min = max(peaks_min_before) if peaks_min_before else np.nan

                # Nearest previous maximum.
                previous_peak_index_max = max(peaks_max_before) if peaks_max_before else np.nan

                # Nearest next minimum.
                next_peak_index_min = min(peaks_min_after) if peaks_min_after else np.nan

                # Nearest next maximum.
                next_peak_index_max = min(peaks_max_after) if peaks_max_after else np.nan

                # Choose the nearest previous extrema, preserving min/max type.
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
                    


                if not pd.isnull(previous_peak_index) and not pd.isnull(next_peak_index):
                    # Total distance between the surrounding extrema.
                    path_len = next_peak_index - previous_peak_index

                    # Remaining distance to the next extrema.
                    remain_len = next_peak_index - position

                    # Fraction of the extrema-to-extrema path still ahead.
                    percentage = remain_len / path_len


                    # Sign the KPI according to whether the previous extrema was a min or max.
                    if(previous_peak_type == 'min'):
                        kpi -= percentage

                    if(previous_peak_type == 'max'):
                        kpi += percentage

                    if weights is not None and len(weights) > 0:

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
                

        
            kpi_series = pd.concat([kpi_series, pd.Series([kpi])], ignore_index=True)
            weigthed_kpi_series = pd.concat([weigthed_kpi_series, pd.Series([weigthed_kpi])], ignore_index=True)









        return kpi_series, weigthed_kpi_series



    # Fitness function
    def genOpt_evaluateMSEFitness(self, individual):

        data = self.data
        last_date = self.genOpt_last_date
        
        print(f'self.genOpt_last_date {self.genOpt_last_date}')
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
                windowing = self.windowing,
                kaiser_beta = self.kaiser_beta,
                linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                period_related_rebuild_range = self.period_related_rebuild_range,
                period_related_rebuild_multiplier = period_related_rebuild_multiplier
                
            )
            


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
                                        windowing = None, # None 'kaiser',
                                        kaiser_beta = 3,
                                        period_related_rebuild_range = False,
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
        self.time_tracking = time_tracking
        self.windowing = windowing
        self.kaiser_beta = kaiser_beta



        # Build optimization domains according to detrending mode.
        
        if(self.period_related_rebuild_range == True): 
            
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
            
            print(f'hp_filter_lambda_min {hp_filter_lambda_min}')
            print(f'hp lambda logarithmic sequence: {logarithmic_sequence}')
            
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
            print(f'multiprocessing.get_start_method = {multiprocessing.get_start_method()}')

            # Use spawn on Windows before registering pool.map.
            if multiprocessing.get_start_method() != 'spawn':
                print('Setting Spawn method')
                multiprocessing.set_start_method('spawn')

            cpu_count = multiprocessing.cpu_count()
            print(f"CPU count: {cpu_count}")

            # Enable multiprocessing for fitness evaluation.
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
        
        else:
            
            print('Multiprocessing disabled')


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
        
        

        # Create the folder if it doesn't exist
        folder_path = os.path.join(os.getcwd(), clean_folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Check if the file already exists
        file_path = os.path.join(folder_path, f"{clean_file_name}.csv")

        if os.path.isfile(file_path):
            # If the file exists, load the existing DataFrame
            existing_dataframe = pd.read_csv(file_path)
            
            existing_dataframe.reset_index(drop=True, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            

            # Concatenate the existing DataFrame with the new data
            combined_dataframe = pd.concat([existing_dataframe, dataframe], ignore_index=True)
        else:
            # If the file doesn't exist, use the new DataFrame
            combined_dataframe = dataframe

        if(update_column == True and update_column_name != None and update_column_value != None):

            if(filter_column_name == None and filter_column_value == None):
                combined_dataframe[update_column_name] = update_column_value

            elif(filter_column_name != None and filter_column_value != None):


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



        return data_df


    def min_max_analysis_concatenated_dataframe(self, #cyPredict_ist,
                                                data_column_name,
                                                current_date,
                                                periods_pars,
                                                population_n = 10,
                                                CXPB = 0.7,
                                                MUTPB = 0.3,
                                                NGEN = 400,
                                                MultiAn_fitness_type = "mse",
                                                MultiAn_fitness_type_svg_smoothed = False,
                                                MultiAn_fitness_type_svg_filter = 4,
                                                reference_detrended_data = "less_detrended", # longest less_detrended
                                                opt_algo_type = 'cpp_genetic_amp_freq_phase', # 'cpp_genetic_amp_freq_phase' 'nlopt_amplitudes_freqs_phases', #'', # 'genetic_omny_frequencies', 'tpe', 'atpe'
                                                amplitudes_inizialization_type = "all_equal_middle_value", # "random", "all_equal_middle_value", "transform_amplitudes"
                                                frequencies_ft = True,
                                                phases_ft = False,                                                                            
                                                detrend_type = 'hp_filter', #hp_filter linear lowess
                                                cut_to_date_before_detrending = True,
                                                linear_filter_window_size_multiplier = 1.85,
                                                period_related_rebuild_range = True,
                                                period_related_rebuild_multiplier = 2,
                                                discretization_steps = 1000,
                                                lowess_k = 6,
                                                windowing = 'kaiser', #'kaiser', # 'kaiser', #kaiser None
                                                kaiser_beta = 1,                                
                                                enabled_multiprocessing = True,
                                                N_elements_prices_CDC = 6,
                                                N_elements_goertzel_CDC = 3,
                                                N_elements_alignmentsKPI_CDC = 10,
                                                N_elements_weigthed_alignmentsKPI_CDCC = 10,
                                                time_tracking = True,
                                                print_activity_remarks = False 
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
        time_tracking, print_activity_remarks : bool
            Runtime/progress logging flags.

        Returns
        -------
        pandas.DataFrame
            A one-row dataframe containing ``datetime``, ``best_fitness_value``,
            OHLCV values, derived price deltas, projected extrema descriptors
            and cycle-alignment trend fields.
        """
        
        


        datetime_df = pd.DataFrame()
        best_fitness_value_df = pd.DataFrame()

        # Recompute the anchor index against self.data for row-level OHLCV fields.
        max_datetime = self.data.index[self.data.index <= current_date].max()
        current_date_idx = self.data.index.get_loc(max_datetime)

        
        

        elaborated_data_df, signals_results_df, composite_signal, configurations, bb_delta, cdc_rsi, index_of_max_time_for_cd, scaled_signals, best_fitness_value = self.multiperiod_analysis(
                                                data_column_name = data_column_name,
                                                current_date = current_date, #'2024-01-18',
                                                periods_pars = periods_pars,
                                                population_n = population_n,
                                                CXPB = CXPB,
                                                MUTPB = MUTPB,
                                                NGEN = NGEN,
                                                MultiAn_fitness_type = MultiAn_fitness_type,
                                                MultiAn_fitness_type_svg_smoothed = MultiAn_fitness_type_svg_smoothed,
                                                MultiAn_fitness_type_svg_filter =MultiAn_fitness_type_svg_filter,
                                                reference_detrended_data = reference_detrended_data, # longest less_detrended
                                                opt_algo_type = opt_algo_type,
                                                amplitudes_inizialization_type = amplitudes_inizialization_type,
                                                frequencies_ft = frequencies_ft,
                                                phases_ft = phases_ft,                                                                            
                                                detrend_type = detrend_type, #hp_filter linear lowess
                                                cut_to_date_before_detrending = cut_to_date_before_detrending,
                                                linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                period_related_rebuild_range = period_related_rebuild_range,
                                                period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                                                discretization_steps = discretization_steps,
                                                lowess_k = lowess_k,
                                                windowing = windowing, #'kaiser', # 'kaiser', #kaiser None
                                                kaiser_beta = kaiser_beta,                                
                                                enabled_multiprocessing = enabled_multiprocessing,
                                                show_charts = False,
                                                time_tracking = time_tracking,
                                                print_activity_remarks = print_activity_remarks
                                            )
        

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
                                                                   current_time_idx = index_of_max_time_for_cd,
                                                                   suffix_col_name = 'scaled_alignmentsKPI_',
                                                                   N_elements = N_elements_alignmentsKPI_CDC)

        min_max_weigthed_alignmentsKPI_CDC_analysis = self.min_max_analysis(data = scaled_signals['scaled_weigthed_alignmentsKPI'],
                                                                            current_time_idx = index_of_max_time_for_cd,
                                                                            suffix_col_name = 'weigthed_alignmentsKPI_',
                                                                            N_elements = N_elements_weigthed_alignmentsKPI_CDCC)

        base_data = pd.DataFrame()
        base_data = pd.DataFrame([self.data.iloc[current_date_idx][['Open', 'Low', 'High', 'Close', 'Volume']].values], columns=['Open', 'Low', 'High', 'Close', 'Volume'])






        base_data['CO'] = base_data['Close'] - base_data['Open']
        base_data['HL'] = base_data['High'] - base_data['Low']
        base_data['CL'] = base_data['Close'] - base_data['Low']
        base_data['CH'] = base_data['Close'] - base_data['High']
        base_data['HO'] = base_data['High'] - base_data['Open']
        base_data['OL'] = base_data['Open'] - base_data['Low']

        base_data['HL_Volume_effort'] = base_data['HL'] / base_data['Volume']


        if(current_date_idx > 0):
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

        # Assemble the wide output row in the same column order used by notebooks.
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
                                retrieve_pars_from_file = False,
                                optimized_pars_filepath = None,
                                min_period = None,
                                max_period = None,
                                population_n = 10,
                                CXPB = 0.7,
                                MUTPB = 0.3,
                                NGEN = 400,
                                resume = False,
                                file_path = '/My Drive',
                                file_name = '/min_max_prices_analysis.csv',
                                opt_algo_type = 'cpp_genetic_amp_freq_phase', # 'cpp_genetic_amp_freq_phase' 'nlopt_amplitudes_freqs_phases', #'', # 'genetic_omny_frequencies', 'tpe', 'atpe'
                                amplitudes_inizialization_type = "all_equal_middle_value", # "random", "all_equal_middle_value", "transform_amplitudes"
                                frequencies_ft = True,
                                phases_ft = False,                                                                            
                                detrend_type = 'hp_filter', #hp_filter linear lowess
                                cut_to_date_before_detrending = True,
                                linear_filter_window_size_multiplier = 1.85,
                                period_related_rebuild_range = True,
                                period_related_rebuild_multiplier = 2,
                                discretization_steps = 1000,
                                lowess_k = 6,
                                windowing = None, #'kaiser', # 'kaiser', #kaiser None
                                kaiser_beta = 1,                                
                                enabled_multiprocessing = True,
                                time_tracking = True,
                                print_activity_remarks = False 
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
        time_tracking, print_activity_remarks : bool
            Runtime/progress logging flags.

        Returns
        -------
        pandas.DataFrame
            The updated min/max analysis dataframe, also written to CSV after
            each appended row.
        """
        
        file_path_name = file_path + file_name
        
        # Resume from an existing CSV only when explicitly requested.
        if resume and os.path.exists(file_path_name):
            min_max_CDC_analysis_df = pd.read_csv(file_path_name, parse_dates=['datetime'])

            print("File CSV esistente caricato.")
        else:
            # Start a new accumulation dataframe.
            min_max_CDC_analysis_df = pd.DataFrame()
            print("Nessun file CSV trovato o resume == False. Creazione di un nuovo dataframe.")
            

        # Limit processing to the requested lookback ending at current_date.
        if pd.to_datetime(current_date) not in self.data.index:
            print(f"Data {current_date} non trovata nei dati. Uscita.")
            return min_max_CDC_analysis_df

        # Select the candidate dates to process.
        start_idx = self.data.index.get_loc(pd.to_datetime(current_date)) - lookback_periods
        if start_idx < 0:
            start_idx = 0
        filtered_data = self.data.iloc[start_idx:self.data.index.get_loc(pd.to_datetime(current_date)) + 1]
        
        
        
        
        # In resume mode, keep only dates not already present in the CSV.
        if not min_max_CDC_analysis_df.empty:            

            # Normalize persisted datetimes before comparing them with self.data.
            min_max_CDC_analysis_df['datetime'] = pd.to_datetime(min_max_CDC_analysis_df['datetime'], errors='coerce')

            # Find dates missing from the persisted output.
            missing_dates = filtered_data.index.difference(min_max_CDC_analysis_df['datetime'])

            # Process only missing rows.
            filtered_data = filtered_data.loc[missing_dates]

        else:
            print("Il file CSV è vuoto, quindi tutte le date sono considerate nuove.")


        # Build and append one wide feature row for each missing date.
        for date in filtered_data.index:
            # Preserve timezone information when formatting the analysis date.

            if date.tzinfo is None:
                date_str = date.replace(tzinfo=pd.Timestamp.utcnow().tz).isoformat()
            else:
                date_str = date.isoformat()
            
            
            print('\n\n----------------------------------------------------------')
            print(f'min_max_analysis_concatenated_dataframe on date {date_str}')
            print('----------------------------------------------------------')
            
            if(retrieve_pars_from_file and optimized_pars_filepath is not None):
                
                cycles_parameters = self.get_most_updated_optimization_pars(optimized_pars_filepath, date_str)

                
                
        
            if(min_period is not None):
                cycles_parameters = cycles_parameters[cycles_parameters['min_period'] >= min_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)

            if(max_period is not None):
                cycles_parameters = cycles_parameters[cycles_parameters['max_period'] <= max_period]
                cycles_parameters = cycles_parameters.reset_index(drop=True)
                    
                
            # Run the single-date min/max feature extraction.
            analyzed_row = self.min_max_analysis_concatenated_dataframe(
                                                  data_column_name = 'Close',
                                                  current_date = date_str, #'2024-01-18',
                                                  periods_pars = cycles_parameters,
                                                  population_n = population_n,
                                                  CXPB = CXPB,
                                                  MUTPB = MUTPB,
                                                  NGEN = NGEN,
                                                  MultiAn_fitness_type = "mse",
                                                  MultiAn_fitness_type_svg_smoothed = False,
                                                  MultiAn_fitness_type_svg_filter = 4,
                                                  reference_detrended_data = "less_detrended", # longest less_detrended
                                                  opt_algo_type = opt_algo_type,
                                                  detrend_type = detrend_type, #hp_filter linear
                                                  linear_filter_window_size_multiplier = linear_filter_window_size_multiplier,
                                                  period_related_rebuild_range = period_related_rebuild_range,
                                                  period_related_rebuild_multiplier = period_related_rebuild_multiplier,
                                                  cut_to_date_before_detrending = cut_to_date_before_detrending,  
                                                  frequencies_ft = frequencies_ft,  
                                                  phases_ft = phases_ft,  
                                                  amplitudes_inizialization_type = amplitudes_inizialization_type,  
                                                  discretization_steps = discretization_steps,  
                                                  lowess_k = lowess_k,  
                                                  windowing = windowing,  
                                                  kaiser_beta = kaiser_beta,  
                                                  enabled_multiprocessing = enabled_multiprocessing,
                                                  time_tracking = time_tracking,
                                                  print_activity_remarks = print_activity_remarks 
            )
            
            if 'datetime' in analyzed_row.columns:
                analyzed_row['datetime'] = pd.to_datetime(analyzed_row['datetime'], errors='coerce')

            # Append the new row to the accumulated dataframe.
            min_max_CDC_analysis_df = pd.concat([min_max_CDC_analysis_df, analyzed_row])

       
            # Normalize timestamps before sorting/persisting.
            min_max_CDC_analysis_df = min_max_CDC_analysis_df.sort_values(by='datetime')
            min_max_CDC_analysis_df = min_max_CDC_analysis_df.drop_duplicates(subset=['datetime'], keep='first')


            # Persist after each row so long runs can resume safely.
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
            

            # Filtra solo le date che sono nel passato o uguali alla data corrente
            group = group[group['analysis_reference_date'] <= current_date]

            if not group.empty:

                # Per ogni optimization_label, calcola la differenza assoluta tra le date e la data corrente
                group['time_diff'] = (current_date - group['analysis_reference_date']).abs()


                # Trova la data più vicina alla data corrente
                closest_date = group.loc[group['time_diff'].idxmin(), 'analysis_reference_date']

                # Seleziona TUTTE le righe con la data più vicina alla data corrente
                closest_group = group[group['analysis_reference_date'] == closest_date]

                # Ora puoi trovare l'indice della riga con il valore minimo di 'best_fitness_value_sum' per ciascun optimization_label
                idx_min_fitness = closest_group.groupby('optimization_label')['best_fitness_value_sum'].idxmin()
                final_selection = closest_group.loc[idx_min_fitness]


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
