"""Structured configuration objects for cyPredict public workflows.

The legacy methods keep their broad keyword signatures for notebook
compatibility. These dataclasses provide a typed, grouped alternative that can
be adopted gradually without changing calculation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    """Constructor-level data loading configuration."""

    data_source: str = "yfinance"
    data_filename: str | None = None
    ticker: str = "SPY"
    data_start_date: str = "2004-01-01"
    data_end_date: str | None = None
    data_timeframe: str = "1d"
    data_storage_path: str = "\\cyPredict\\"

    def to_constructor_kwargs(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source,
            "data_filename": self.data_filename,
            "ticker": self.ticker,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "data_timeframe": self.data_timeframe,
            "data_storage_path": self.data_storage_path,
        }


@dataclass
class DetrendConfig:
    """Detrending controls shared by single-range and multiperiod analysis."""

    detrend_type: str = "hp_filter"
    detrend_window: int = 0
    hp_filter_lambda: float = 100
    jp_filter_p: int = 4
    jp_filter_h: int = 8
    cut_to_date_before_detrending: bool = True
    lowess_k: int = 3
    linear_filter_window_size_multiplier: float = 1

    def to_analysis_kwargs(self) -> dict[str, Any]:
        return {
            "detrend_type": self.detrend_type,
            "detrend_window": self.detrend_window,
            "hp_filter_lambda": self.hp_filter_lambda,
            "jp_filter_p": self.jp_filter_p,
            "jp_filter_h": self.jp_filter_h,
            "cut_to_date_before_detrending": self.cut_to_date_before_detrending,
            "lowess_k": self.lowess_k,
        }

    def to_multiperiod_kwargs(self) -> dict[str, Any]:
        return {
            "detrend_type": self.detrend_type,
            "cut_to_date_before_detrending": self.cut_to_date_before_detrending,
            "lowess_k": self.lowess_k,
            "linear_filter_window_size_multiplier": self.linear_filter_window_size_multiplier,
        }


@dataclass
class GoertzelConfig:
    """Goertzel spectrum and dominant-period selection configuration."""

    transform_precision: float = 0.01
    final_kept_n_dominant_circles: int = 1
    dominant_cicles_sorting_type: str = "global_score"
    limit_n_harmonics: int | None = None
    min_period: int = 20
    max_period: int = 100
    bartel_peaks_filtering: bool = True
    bartel_scoring_threshold: float = 0.5
    windowing: str | None = None
    kaiser_beta: float = 5

    def to_analysis_kwargs(self) -> dict[str, Any]:
        return {
            "transform_precision": self.transform_precision,
            "final_kept_n_dominant_circles": self.final_kept_n_dominant_circles,
            "dominant_cicles_sorting_type": self.dominant_cicles_sorting_type,
            "limit_n_harmonics": self.limit_n_harmonics,
            "min_period": self.min_period,
            "max_period": self.max_period,
            "bartel_peaks_filtering": self.bartel_peaks_filtering,
            "bartel_scoring_threshold": self.bartel_scoring_threshold,
            "windowing": self.windowing,
            "kaiser_beta": self.kaiser_beta,
        }

    def to_multiperiod_kwargs(self) -> dict[str, Any]:
        return {
            "windowing": self.windowing,
            "kaiser_beta": self.kaiser_beta,
        }


@dataclass
class OptimizationConfig:
    """Optimization controls for multiperiod reconstruction."""

    population_n: int = 40
    CXPB: float = 0.7
    MUTPB: float = 0.3
    NGEN: int = 100
    MultiAn_fitness_type: str = "mse"
    MultiAn_fitness_type_svg_smoothed: bool = True
    MultiAn_fitness_type_svg_filter: int = 5
    weigth: float = -1.0
    opt_algo_type: str = "genetic_omny_frequencies"
    amplitudes_inizialization_type: str = "random"
    frequencies_ft: bool = True
    phases_ft: bool = True
    discretization_steps: int = 1000
    enabled_multiprocessing: bool = True
    random_seed: int | None = None

    def to_multiperiod_kwargs(self) -> dict[str, Any]:
        return {
            "population_n": self.population_n,
            "CXPB": self.CXPB,
            "MUTPB": self.MUTPB,
            "NGEN": self.NGEN,
            "MultiAn_fitness_type": self.MultiAn_fitness_type,
            "MultiAn_fitness_type_svg_smoothed": self.MultiAn_fitness_type_svg_smoothed,
            "MultiAn_fitness_type_svg_filter": self.MultiAn_fitness_type_svg_filter,
            "weigth": self.weigth,
            "opt_algo_type": self.opt_algo_type,
            "amplitudes_inizialization_type": self.amplitudes_inizialization_type,
            "frequencies_ft": self.frequencies_ft,
            "phases_ft": self.phases_ft,
            "discretization_steps": self.discretization_steps,
            "enabled_multiprocessing": self.enabled_multiprocessing,
            "random_seed": self.random_seed,
        }


@dataclass
class ProjectionConfig:
    """Controls for projection range and reference signal selection."""

    best_fit_start_back_period: int | None = None
    reference_detrended_data: str = "longest"
    enable_cycles_alignment_analysis: bool = True
    period_related_rebuild_range: bool = False
    period_related_rebuild_multiplier: float = 2.5

    def to_multiperiod_kwargs(self) -> dict[str, Any]:
        return {
            "best_fit_start_back_period": self.best_fit_start_back_period,
            "reference_detrended_data": self.reference_detrended_data,
            "enable_cycles_alignment_analysis": self.enable_cycles_alignment_analysis,
            "period_related_rebuild_range": self.period_related_rebuild_range,
            "period_related_rebuild_multiplier": self.period_related_rebuild_multiplier,
        }


@dataclass
class OutputConfig:
    """Output, chart and structured logging controls."""

    show_charts: bool = False
    print_report: bool = True
    centered_averages: bool = True
    other_correlations: bool = False
    debug: bool = False
    log_level: str | None = None
    log_to_console: bool | None = None
    log_to_file: bool | None = None
    log_dir: str | None = None
    log_run_id: str | None = None

    def to_analysis_kwargs(self) -> dict[str, Any]:
        return {
            "centered_averages": self.centered_averages,
            "other_correlations": self.other_correlations,
            "show_charts": self.show_charts,
            "print_report": self.print_report,
            "debug": self.debug,
        }

    def to_multiperiod_kwargs(self) -> dict[str, Any]:
        return {
            "show_charts": self.show_charts,
            "log_level": self.log_level,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
            "log_dir": self.log_dir,
            "log_run_id": self.log_run_id,
        }


@dataclass
class AnalysisConfig:
    """Structured configuration for ``analyze_and_plot``."""

    data: Any = None
    data_column_name: str = "Close"
    num_samples: int | None = None
    start_date: Any = None
    current_date: Any = None
    goertzel: GoertzelConfig = field(default_factory=GoertzelConfig)
    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_legacy_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "data": self.data,
            "data_column_name": self.data_column_name,
            "num_samples": self.num_samples,
            "start_date": self.start_date,
            "current_date": self.current_date,
        }
        kwargs.update(self.goertzel.to_analysis_kwargs())
        kwargs.update(self.detrend.to_analysis_kwargs())
        kwargs.update(self.output.to_analysis_kwargs())
        return kwargs


@dataclass
class MultiPeriodAnalysisConfig:
    """Structured configuration for ``multiperiod_analysis``."""

    data_column_name: str
    current_date: Any
    periods_pars: Any
    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    goertzel: GoertzelConfig = field(default_factory=GoertzelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_legacy_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "data_column_name": self.data_column_name,
            "current_date": self.current_date,
            "periods_pars": self.periods_pars,
        }
        kwargs.update(self.projection.to_multiperiod_kwargs())
        kwargs.update(self.output.to_multiperiod_kwargs())
        kwargs.update(self.optimization.to_multiperiod_kwargs())
        kwargs.update(self.goertzel.to_multiperiod_kwargs())
        kwargs.update(self.detrend.to_multiperiod_kwargs())
        return kwargs
