"""Public cyPredict class composed from refactored mixins."""

from .native_imports import (
    REQUIRED_CYGAOPT_ABI_VERSION,
    ensure_native_module_paths,
    require_native_abi,
)

ensure_native_module_paths()

import cyGAopt
import cyGAoptMultiCore
import plotly.io as pio

require_native_abi(cyGAopt, "cyGAopt", REQUIRED_CYGAOPT_ABI_VERSION)
require_native_abi(cyGAoptMultiCore, "cyGAoptMultiCore", REQUIRED_CYGAOPT_ABI_VERSION)

# Keep the historical notebook renderer default.
pio.renderers.default = "iframe"

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


# Preserve both historical calling styles:
# - from cyPredict import cyPredict; cyPredict(...)
# - from cyPredict import cyPredict; cyPredict.cyPredict(...)
cyPredict.cyPredict = cyPredict
