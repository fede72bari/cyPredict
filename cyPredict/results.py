"""Result objects for cyPredict public workflows.

The legacy APIs still return tuples/dataframes. These objects provide named
fields and explicit legacy conversion for callers that want a clearer contract
without changing the calculation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """Named result for ``analyze_and_plot``."""

    current_date: Any
    index_of_max_time_for_cd: Any
    original_data: Any
    signals_results: Any
    configuration: Any
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_legacy_tuple(cls, result: tuple[Any, ...]) -> "AnalysisResult":
        """Build an ``AnalysisResult`` from the five-value legacy tuple."""
        if len(result) != 5:
            raise ValueError(f"analyze_and_plot returned {len(result)} values; expected 5")
        ok = any(value is not None for value in result)
        return cls(*result, ok=ok)

    def as_legacy_tuple(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return the exact tuple shape produced by ``analyze_and_plot``."""
        return (
            self.current_date,
            self.index_of_max_time_for_cd,
            self.original_data,
            self.signals_results,
            self.configuration,
        )

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        """Return metadata, optionally including large dataframe-like values."""
        payload = {
            "ok": self.ok,
            "current_date": self.current_date,
            "index_of_max_time_for_cd": self.index_of_max_time_for_cd,
            "warnings": list(self.warnings),
        }
        if include_data:
            payload.update(
                {
                    "original_data": self.original_data,
                    "signals_results": self.signals_results,
                    "configuration": self.configuration,
                }
            )
        return payload


@dataclass
class MultiPeriodResult:
    """Named result for ``multiperiod_analysis``."""

    elaborated_data_series: Any
    signals_results_series: Any
    composite_signal: Any
    configurations_series: Any
    bb_delta: Any
    cdc_rsi: Any
    index_of_max_time_for_cd: Any = None
    scaled_signals: Any = None
    best_fitness_value: Any = None
    ok: bool = True
    warnings: list[str] = field(default_factory=list)
    legacy_length: int = 9

    @classmethod
    def from_legacy_tuple(cls, result: tuple[Any, ...]) -> "MultiPeriodResult":
        """Build a result object from the legacy six- or nine-value tuple."""
        if len(result) not in (6, 9):
            raise ValueError(f"multiperiod_analysis returned {len(result)} values; expected 6 or 9")
        values = tuple(result) + (None,) * (9 - len(result))
        ok = any(value is not None for value in result)
        return cls(*values[:9], ok=ok, legacy_length=len(result))

    def as_legacy_tuple(self) -> tuple[Any, ...]:
        """Return the original six- or nine-value legacy tuple shape."""
        values = (
            self.elaborated_data_series,
            self.signals_results_series,
            self.composite_signal,
            self.configurations_series,
            self.bb_delta,
            self.cdc_rsi,
            self.index_of_max_time_for_cd,
            self.scaled_signals,
            self.best_fitness_value,
        )
        return values[: self.legacy_length]

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        """Return metadata, optionally including large analysis outputs."""
        payload = {
            "ok": self.ok,
            "index_of_max_time_for_cd": self.index_of_max_time_for_cd,
            "best_fitness_value": self.best_fitness_value,
            "warnings": list(self.warnings),
            "legacy_length": self.legacy_length,
        }
        if include_data:
            payload.update(
                {
                    "elaborated_data_series": self.elaborated_data_series,
                    "signals_results_series": self.signals_results_series,
                    "composite_signal": self.composite_signal,
                    "configurations_series": self.configurations_series,
                    "bb_delta": self.bb_delta,
                    "cdc_rsi": self.cdc_rsi,
                    "scaled_signals": self.scaled_signals,
                }
            )
        return payload


@dataclass
class MinMaxAnalysisResult:
    """Named result for min/max dataframe workflows."""

    dataframe: Any
    ok: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_legacy_value(cls, dataframe: Any) -> "MinMaxAnalysisResult":
        """Wrap a legacy min/max dataframe return value."""
        return cls(dataframe=dataframe, ok=dataframe is not None)

    def as_legacy_value(self) -> Any:
        """Return the raw dataframe value expected by legacy callers."""
        return self.dataframe

    def as_legacy_tuple(self) -> tuple[Any]:
        """Return a one-value tuple containing the dataframe."""
        return (self.dataframe,)

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        """Return result metadata, optionally including the dataframe."""
        payload = {
            "ok": self.ok,
            "warnings": list(self.warnings),
        }
        if include_data:
            payload["dataframe"] = self.dataframe
        return payload
