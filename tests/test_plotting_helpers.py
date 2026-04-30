from unittest.mock import patch

import numpy as np
import pandas as pd

from cyPredict.core.plotting import PlottingMixin


class PlottingProbe(PlottingMixin):
    ticker = "QQQ"

    def log_debug(self, *args, **kwargs):
        return None

    def log_warning(self, *args, **kwargs):
        return None


def test_single_range_plotting_helper_builds_legacy_figures():
    probe = PlottingProbe()
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    original_data = pd.DataFrame(
        {
            "Close": np.linspace(100, 105, len(index)),
            "detrended": np.linspace(-1, 1, len(index)),
            "composite_dominant_circles_signal": np.sin(np.arange(len(index))),
        },
        index=index,
    )

    with patch("plotly.basedatatypes.BaseFigure.show"):
        spectrum, figure = probe.plot_single_range_analysis_charts(
            frequency_range=[0.1, 0.2, 0.3],
            harmonics_amplitudes=[1.0, 2.0, 1.0],
            original_data=original_data,
            data_column_name="Close",
            index_of_max_time_for_cd=index[3],
        )

    assert spectrum.layout.title.text == "Frequency Spectrum"
    assert figure.layout.title.text == "Goertzel Dominant Cyrcles Analysis"
    assert len(figure.data) == 3


def test_multiperiod_plotting_helper_builds_projection_figure():
    probe = PlottingProbe()
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    reduced_data = pd.DataFrame({"Close": np.linspace(100, 120, len(index))}, index=index)
    composite_signal = pd.DataFrame({"composite_signal": np.sin(np.arange(len(index)))}, index=index)
    elaborated_data_series = [pd.DataFrame(index=index)]

    with patch("plotly.basedatatypes.BaseFigure.show"), patch("cyPredict.core.plotting.plot"):
        figure = probe.plot_multiperiod_analysis_charts(
            reduced_data=reduced_data,
            data_column_name="Close",
            composite_signal=composite_signal,
            elaborated_data_series=elaborated_data_series,
            max_length_series_index=0,
            scaled_composite_signal=np.sin(np.arange(len(index))),
            scaled_goertzel_composite_signal=np.cos(np.arange(len(index))),
            scaled_detrended=np.linspace(-1, 1, len(index)),
            scaled_alignmentsKPI=np.linspace(0, 1, len(index)),
            scaled_weigthed_alignmentsKPI=np.linspace(1, 0, len(index)),
            index_of_max_time_for_cd=12,
        )

    assert figure.layout.title.text == "Goertzel Dominant Cyrcles Analysis"
    assert len(figure.data) == 6
