"""Legacy public package interface for cyPredict.

The implementation lives in :mod:`cyPredict.cypredict`.  This module keeps the
historical import contracts stable while the codebase is split into smaller
modules.
"""

import sys

if __name__ == "cypredict" and hasattr(sys.modules.get("cyPredict.cyPredict"), "__all__"):
    _canonical = sys.modules["cyPredict.cyPredict"]
    __all__ = list(_canonical.__all__)
    globals().update({name: getattr(_canonical, name) for name in __all__})
    sys.modules[__name__] = _canonical
elif __name__ == "cypredict" and hasattr(sys.modules.get("cyPredict"), "__all__"):
    _canonical = sys.modules["cyPredict"]
    __all__ = list(_canonical.__all__)
    globals().update({name: getattr(_canonical, name) for name in __all__})
    sys.modules[__name__] = _canonical
else:
    if __name__ == "cypredict":
        sys.modules.setdefault("cyPredict.cyPredict", sys.modules[__name__])
    elif __name__ == "cyPredict.cyPredict":
        sys.modules.setdefault("cypredict", sys.modules[__name__])

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
    from .results import AnalysisResult, MinMaxAnalysisResult, MultiPeriodResult

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
