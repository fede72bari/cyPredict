import ast
import inspect
from pathlib import Path

from cyPredict.core.analysis import AnalysisMixin
from cyPredict.core.extrema import ExtremaMixin
from cyPredict.core.indicators import IndicatorsMixin
from cyPredict.core.minmax import MinMaxMixin
from cyPredict.core.multiperiod import MultiperiodMixin
from cyPredict.core.reconstruction import ReconstructionMixin
from cyPredict.core.state import StateMixin


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "cyPredict" / "core"

ALLOWED_UNUSED_PARAMETERS = {
    # NLopt callback signature: the optimizer passes ``grad`` even when the
    # objective does not use gradients.
    ("optimization.py", "loss", "grad"),
}


def _function_parameters(node):
    params = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
    if node.args.vararg is not None:
        params.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        params.append(node.args.kwarg.arg)
    return [param for param in params if param != "self"]


def _names_used_by_function(node):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_FunctionDef(self, child):
            if child is node:
                for statement in child.body:
                    self.visit(statement)

        def visit_Lambda(self, child):
            return None

        def visit_Name(self, child):
            self.names.add(child.id)

    visitor = Visitor()
    visitor.visit(node)
    return visitor.names


def _iter_functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            yield node


def test_core_functions_do_not_have_unreviewed_unused_parameters():
    offenders = []

    for path in sorted(CORE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in _iter_functions(tree):
            used_names = _names_used_by_function(function)
            for parameter in _function_parameters(function):
                key = (path.name, function.name, parameter)
                if parameter not in used_names and key not in ALLOWED_UNUSED_PARAMETERS:
                    offenders.append(f"{path.name}:{function.lineno}:{function.name}.{parameter}")

    assert offenders == []


def _assert_absent(callable_obj, removed_parameters):
    parameters = inspect.signature(callable_obj).parameters
    for parameter in removed_parameters:
        assert parameter not in parameters


def test_removed_legacy_parameters_are_not_reintroduced():
    _assert_absent(
        StateMixin.__init__,
        {"time_tracking", "print_activity_remarks"},
    )
    _assert_absent(
        AnalysisMixin.analyze_and_plot,
        {
            "include_calibrated_MACD",
            "include_calibrated_RSI",
            "indicators_signal_calcualtion",
            "enabled_multiprocessing",
        },
    )
    _assert_absent(
        MultiperiodMixin.multiperiod_analysis,
        {
            "pars_from_opt_file",
            "files_path_name",
            "bb_delta_fixed_periods",
            "bb_delta_sg_filter_window",
            "RSI_cycles_analysis_type",
            "time_zone",
            "time_tracking",
            "print_activity_remarks",
        },
    )
    _assert_absent(
        MinMaxMixin.min_max_analysis_concatenated_dataframe,
        {
            "pars_from_opt_file",
            "files_path_name",
            "bb_delta_fixed_periods",
            "bb_delta_sg_filter_window",
            "RSI_cycles_analysis_type",
            "show_charts",
        },
    )
    _assert_absent(
        MinMaxMixin.get_min_max_analysis_df,
        {"source_type", "data_column_name", "GoogleDriveMountPoint", "index_column_name"},
    )
    _assert_absent(
        IndicatorsMixin.indict_MACD_SGMACD,
        {"signals_results"},
    )
    _assert_absent(
        IndicatorsMixin.indict_RSI_SG_smooth_RSI,
        {"signals_results"},
    )
    _assert_absent(
        IndicatorsMixin.indict_centered_average_deltas,
        {"signals_results"},
    )
    _assert_absent(
        ReconstructionMixin.rebuilt_signal_zeros,
        {"debug"},
    )
    _assert_absent(
        ExtremaMixin.CDC_vs_detrended_correlation,
        {"data", "lowess_k", "best_fit_start_back_period"},
    )
    _assert_absent(
        ExtremaMixin.CDC_vs_detrended_correlation_sum,
        {"best_fit_start_back_period"},
    )
    _assert_absent(
        ExtremaMixin.trade_predicted_dominant_cicles_peaks,
        {"data"},
    )
