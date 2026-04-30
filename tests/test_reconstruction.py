import numpy as np
import pandas as pd

from cyPredict.core.reconstruction import ReconstructionMixin


class _ReconstructionProbe(ReconstructionMixin):
    pass


def test_composite_signal_uses_best_frequency_and_phase_when_available():
    probe = _ReconstructionProbe()
    cycles = pd.DataFrame(
        {
            "peak_periods": [4.0],
            "peak_frequencies": [0.0],
            "peak_phases": [0.0],
            "best_frequencies": [0.25],
            "best_phases": [0.0],
            "start_rebuilt_signal_index": [0],
        }
    )

    result = probe.cicles_composite_signals(4, [1.0], cycles, range(4), "composite")

    np.testing.assert_allclose(result["composite"].to_numpy(), [0.0, 1.0, 0.0, -1.0], atol=1e-12)


def test_composite_signal_falls_back_to_peak_frequency_and_phase():
    probe = _ReconstructionProbe()
    cycles = pd.DataFrame(
        {
            "peak_periods": [4.0],
            "peak_frequencies": [0.25],
            "peak_phases": [0.0],
            "start_rebuilt_signal_index": [0],
        }
    )

    result = probe.cicles_composite_signals(4, [1.0], cycles, range(4), "composite")

    np.testing.assert_allclose(result["composite"].to_numpy(), [0.0, 1.0, 0.0, -1.0], atol=1e-12)
