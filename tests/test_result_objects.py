import pandas as pd

from cyPredict import AnalysisResult, MinMaxAnalysisResult, MultiPeriodResult
from cyPredict.core.analysis import AnalysisMixin
from cyPredict.core.minmax import MinMaxMixin
from cyPredict.core.multiperiod import MultiperiodMixin


class AnalysisResultProbe(AnalysisMixin):
    def analyze_and_plot(self, *args, **kwargs):
        return ("2026-04-21", 10, "original", "signals", {"min_period": 8})


class MultiperiodResultProbe(MultiperiodMixin):
    def multiperiod_analysis(self, *args, **kwargs):
        return ("elaborated", "signals", "composite", "configs", None, None, 10, "scaled", 0.123)


class MinMaxResultProbe(MinMaxMixin):
    def min_max_analysis_concatenated_dataframe(self, *args, **kwargs):
        return pd.DataFrame([{"best_fitness_value": 0.123}])

    def get_min_max_analysis_df(self, *args, **kwargs):
        return pd.DataFrame([{"analysis_reference_date": "2026-04-21"}])


def test_analysis_result_preserves_legacy_tuple():
    legacy = ("2026-04-21", 10, "original", "signals", {"min_period": 8})
    result = AnalysisResult.from_legacy_tuple(legacy)

    assert result.ok is True
    assert result.current_date == "2026-04-21"
    assert result.as_legacy_tuple() == legacy
    assert result.to_dict()["index_of_max_time_for_cd"] == 10


def test_analysis_result_wrapper_uses_legacy_calculation_path():
    result = AnalysisResultProbe().analyze_and_plot_result()

    assert isinstance(result, AnalysisResult)
    assert result.as_legacy_tuple() == ("2026-04-21", 10, "original", "signals", {"min_period": 8})


def test_multiperiod_result_preserves_success_and_failure_legacy_tuples():
    success = ("elaborated", "signals", "composite", "configs", None, None, 10, "scaled", 0.123)
    success_result = MultiPeriodResult.from_legacy_tuple(success)

    assert success_result.ok is True
    assert success_result.best_fitness_value == 0.123
    assert success_result.as_legacy_tuple() == success

    failure = (None, None, None, None, None, None)
    failure_result = MultiPeriodResult.from_legacy_tuple(failure)

    assert failure_result.ok is False
    assert failure_result.legacy_length == 6
    assert failure_result.as_legacy_tuple() == failure


def test_multiperiod_result_wrapper_uses_legacy_calculation_path():
    result = MultiperiodResultProbe().multiperiod_analysis_result()

    assert isinstance(result, MultiPeriodResult)
    assert result.best_fitness_value == 0.123
    assert result.as_legacy_tuple()[-1] == 0.123


def test_minmax_result_preserves_dataframe_legacy_value():
    dataframe = pd.DataFrame([{"best_fitness_value": 0.123}])
    result = MinMaxAnalysisResult.from_legacy_value(dataframe)

    assert result.ok is True
    assert result.as_legacy_value() is dataframe
    assert result.as_legacy_tuple()[0] is dataframe


def test_minmax_result_wrappers_use_legacy_calculation_path():
    probe = MinMaxResultProbe()

    concatenated = probe.min_max_analysis_concatenated_dataframe_result()
    incremental = probe.get_min_max_analysis_df_result()

    assert isinstance(concatenated, MinMaxAnalysisResult)
    assert isinstance(incremental, MinMaxAnalysisResult)
    assert concatenated.as_legacy_value().iloc[0]["best_fitness_value"] == 0.123
    assert incremental.as_legacy_value().iloc[0]["analysis_reference_date"] == "2026-04-21"
