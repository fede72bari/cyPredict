"""Legacy public package interface for cyPredict.

The implementation lives in :mod:`cyPredict.cypredict`.  This module keeps the
historical import contracts stable while the codebase is split into smaller
modules.
"""

from .config import (
    AnalysisConfig,
    DataConfig,
    DetrendConfig,
    GoertzelConfig,
    MultiPeriodAnalysisConfig,
    OptimizationConfig,
    OutputConfig,
    ProjectionConfig,
)
from .cypredict import cyPredict

__all__ = [
    "AnalysisConfig",
    "DataConfig",
    "DetrendConfig",
    "GoertzelConfig",
    "MultiPeriodAnalysisConfig",
    "OptimizationConfig",
    "OutputConfig",
    "ProjectionConfig",
    "cyPredict",
]
