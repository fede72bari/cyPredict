import pandas as pd

from cyPredict.config import (
    AnalysisConfig,
    DetrendConfig,
    GoertzelConfig,
    MultiPeriodAnalysisConfig,
    OptimizationConfig,
    OutputConfig,
    ProjectionConfig,
)
from cyPredict.core.analysis import AnalysisMixin
from cyPredict.core.multiperiod import MultiperiodMixin


class AnalysisProbe(AnalysisMixin):
    def analyze_and_plot(self, **kwargs):
        return kwargs


class MultiperiodProbe(MultiperiodMixin):
    def multiperiod_analysis(self, **kwargs):
        return kwargs


def test_analysis_config_expands_to_legacy_kwargs():
    config = AnalysisConfig(
        data_column_name="Close",
        current_date="2026-04-21",
        num_samples=128,
        goertzel=GoertzelConfig(min_period=8, max_period=16, kaiser_beta=2),
        detrend=DetrendConfig(detrend_type="hp_filter", hp_filter_lambda=65),
        output=OutputConfig(show_charts=False, print_report=False),
    )

    kwargs = AnalysisProbe().analyze_and_plot_from_config(config)

    assert kwargs["data_column_name"] == "Close"
    assert kwargs["current_date"] == "2026-04-21"
    assert kwargs["num_samples"] == 128
    assert kwargs["min_period"] == 8
    assert kwargs["max_period"] == 16
    assert kwargs["kaiser_beta"] == 2
    assert kwargs["detrend_type"] == "hp_filter"
    assert kwargs["hp_filter_lambda"] == 65
    assert kwargs["show_charts"] is False
    assert kwargs["print_report"] is False


def test_multiperiod_config_expands_to_legacy_kwargs():
    periods = pd.DataFrame(
        [{"num_samples": 128, "final_kept_n_dominant_circles": 8, "min_period": 8, "max_period": 16, "hp_filter_lambda": 65}]
    )
    config = MultiPeriodAnalysisConfig(
        data_column_name="Close",
        current_date="2026-04-21",
        periods_pars=periods,
        detrend=DetrendConfig(detrend_type="hp_filter", linear_filter_window_size_multiplier=1.85),
        goertzel=GoertzelConfig(windowing="kaiser", kaiser_beta=1),
        optimization=OptimizationConfig(
            population_n=100,
            NGEN=4000,
            opt_algo_type="cpp_genetic_amp_freq_phase",
            frequencies_ft=True,
            phases_ft=False,
            random_seed=123,
        ),
        projection=ProjectionConfig(
            reference_detrended_data="less_detrended",
            period_related_rebuild_range=True,
            period_related_rebuild_multiplier=1.5,
        ),
        output=OutputConfig(show_charts=False, log_level="INFO"),
    )

    kwargs = MultiperiodProbe().multiperiod_analysis_from_config(config)

    assert kwargs["data_column_name"] == "Close"
    assert kwargs["current_date"] == "2026-04-21"
    assert kwargs["periods_pars"] is periods
    assert kwargs["population_n"] == 100
    assert kwargs["NGEN"] == 4000
    assert kwargs["opt_algo_type"] == "cpp_genetic_amp_freq_phase"
    assert kwargs["frequencies_ft"] is True
    assert kwargs["phases_ft"] is False
    assert kwargs["random_seed"] == 123
    assert kwargs["reference_detrended_data"] == "less_detrended"
    assert kwargs["period_related_rebuild_range"] is True
    assert kwargs["period_related_rebuild_multiplier"] == 1.5
    assert kwargs["detrend_type"] == "hp_filter"
    assert kwargs["linear_filter_window_size_multiplier"] == 1.85
    assert kwargs["windowing"] == "kaiser"
    assert kwargs["kaiser_beta"] == 1
    assert kwargs["show_charts"] is False
    assert kwargs["log_level"] == "INFO"
