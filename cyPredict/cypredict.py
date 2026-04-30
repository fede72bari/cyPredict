# Basics

from .native_imports import (
    REQUIRED_CYGAOPT_ABI_VERSION,
    ensure_native_module_paths,
    require_native_abi,
)

ensure_native_module_paths()

from goertzel import goertzel_general_shortened as goertzel_general_shortened
from goertzel import goertzel_DFT as goertzel_DFT

from cyfitness import evaluate_fitness
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

require_native_abi(cyGAopt, "cyGAopt", REQUIRED_CYGAOPT_ABI_VERSION)
require_native_abi(cyGAoptMultiCore, "cyGAoptMultiCore", REQUIRED_CYGAOPT_ABI_VERSION)

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


from .core.analysis import AnalysisMixin
from .core.data import DataMixin
from .core.dates import DatesMixin
from .core.detrending import DetrendingMixin
from .core.diagnostics import DiagnosticsMixin
from .core.extrema import ExtremaMixin
from .core.indicators import IndicatorsMixin
from .core.minmax import MinMaxMixin
from .core.multiperiod import MultiperiodMixin
from .core.optimization import OptimizationMixin
from .core.persistence import PersistenceMixin
from .core.plotting import PlottingMixin
from .core.reconstruction import ReconstructionMixin
from .core.scoring import ScoringMixin
from .core.spectral import SpectralMixin
from .core.state import StateMixin



class cyPredict(
    AnalysisMixin,
    StateMixin,
    DataMixin,
    DatesMixin,
    DetrendingMixin,
    SpectralMixin,
    DiagnosticsMixin,
    ExtremaMixin,
    IndicatorsMixin,
    MinMaxMixin,
    MultiperiodMixin,
    OptimizationMixin,
    PersistenceMixin,
    PlottingMixin,
    ReconstructionMixin,
    ScoringMixin,
):
    """Cycle-analysis engine for financial time series.

    The class downloads or loads OHLCV data, estimates dominant periods with
    Goertzel-based analysis, reconstructs projected cycle signals, and exposes
    helper workflows used by the research notebooks under
    ``D:\\Dropbox\\TRADING\\STUDIES DEVELOPMENT\\CYCLES ANALYSIS``.

    The public API is still legacy-compatible and intentionally broad. Many
    arguments are meaningful only for selected workflows; those mode-specific
    relationships are documented on the methods where they are consumed.
    """


# Backward compatibility for notebooks that do:
# ``from cyPredict import cyPredict`` and then call ``cyPredict.cyPredict(...)``.
cyPredict.cyPredict = cyPredict


# Preserve both historical calling styles:
# - from cyPredict import cyPredict; cyPredict(...)
# - from cyPredict import cyPredict; cyPredict.cyPredict(...)
cyPredict.cyPredict = cyPredict
