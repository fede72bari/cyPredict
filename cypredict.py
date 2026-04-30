"""Lowercase compatibility facade for the cyPredict package.

The historical import path is ``from cyPredict import cyPredict``.  This module
adds the conventional lowercase import path without moving the package on disk,
which keeps Windows notebooks and existing relative imports stable.
"""

from cyPredict import (
    AnalysisConfig,
    AnalysisResult,
    DataConfig,
    DetrendConfig,
    GoertzelConfig,
    MinMaxAnalysisResult,
    MultiPeriodAnalysisConfig,
    MultiPeriodResult,
    OptimizationConfig,
    OutputConfig,
    ProjectionConfig,
    cyPredict,
)

CyPredict = cyPredict

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "CyPredict",
    "DataConfig",
    "DetrendConfig",
    "GoertzelConfig",
    "MinMaxAnalysisResult",
    "MultiPeriodAnalysisConfig",
    "MultiPeriodResult",
    "OptimizationConfig",
    "OutputConfig",
    "ProjectionConfig",
    "cyPredict",
]
