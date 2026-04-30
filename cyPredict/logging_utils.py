"""Structured logging utilities for cyPredict.

This module is intentionally independent from the heavy calculation module so it
can be tested quickly and adopted progressively.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}

VALID_CATEGORIES = {
    "processing",
    "debug",
    "timing",
    "calculation",
    "data",
    "io",
    "optimization",
    "warning",
    "error",
}


def _safe_token(value: Any, fallback: str) -> str:
    """Return a filesystem-safe token for structured log filenames."""
    text = str(value or fallback)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return text or fallback


def _json_safe(value: Any) -> Any:
    """Convert common Python objects to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class LogEvent:
    """One structured cyPredict log event.

    Attributes store both human-readable fields and machine-readable context.
    ``to_console_line`` provides the compact notebook/console format, while
    ``to_dict`` is used for JSONL persistence.
    """

    timestamp: str
    run_id: str
    level: str
    category: str
    function: str
    message: str
    elapsed_seconds: float
    ticker: str | None = None
    timeframe: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation of the event."""
        data = asdict(self)
        data["context"] = {str(key): _json_safe(value) for key, value in self.context.items()}
        return data

    def to_console_line(self) -> str:
        """Render the event as a single structured console line."""
        parts = [
            self.timestamp,
            self.level,
            self.category,
            self.function,
            f"elapsed={self.elapsed_seconds:.6f}s",
            self.message,
        ]
        if self.ticker:
            parts.insert(3, f"ticker={self.ticker}")
        if self.timeframe:
            parts.insert(4, f"timeframe={self.timeframe}")
        return " | ".join(parts)


class CyPredictLogger:
    """Small structured logger for console and file outputs."""

    def __init__(
        self,
        *,
        ticker: str | None = None,
        timeframe: str | None = None,
        log_dir: str | Path = "logs",
        run_id: str | None = None,
        min_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = False,
    ) -> None:
        """Create a structured logger for one analysis run.

        Parameters
        ----------
        ticker, timeframe : str, optional
            Context copied into each event and into generated filenames.
        log_dir : str or Path, default "logs"
            Directory used only when ``log_to_file`` is true.
        run_id : str, optional
            Stable run identifier. A short random id is generated when omitted.
        min_level : {"DEBUG", "INFO", "WARNING", "ERROR"}, default "INFO"
            Minimum level to emit.
        log_to_console : bool, default True
            Print formatted log lines to stdout.
        log_to_file : bool, default False
            Persist both text ``.log`` and machine-readable ``.jsonl`` files.
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.min_level = min_level.upper()
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.started_at = time.perf_counter()
        self.created_at = datetime.now(timezone.utc)
        self.log_dir = Path(log_dir)
        self._log_path: Path | None = None
        self._jsonl_path: Path | None = None

        if self.min_level not in LEVELS:
            raise ValueError(f"Unsupported log level: {min_level}")

        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            ticker_token = _safe_token(self.ticker, "NA")
            timeframe_token = _safe_token(self.timeframe, "NA")
            base = f"cypredict_{ticker_token}_{timeframe_token}_{stamp}_{self.run_id}"
            self._log_path = self.log_dir / f"{base}.log"
            self._jsonl_path = self.log_dir / f"{base}.jsonl"

    @property
    def log_path(self) -> Path | None:
        """Path to the text log file, if file logging is enabled."""
        return self._log_path

    @property
    def jsonl_path(self) -> Path | None:
        """Path to the JSONL log file, if file logging is enabled."""
        return self._jsonl_path

    def is_enabled(self, level: str) -> bool:
        """Return whether ``level`` passes the configured minimum level."""
        level_name = level.upper()
        if level_name not in LEVELS:
            raise ValueError(f"Unsupported log level: {level}")
        return LEVELS[level_name] >= LEVELS[self.min_level]

    def emit(
        self,
        message: str,
        *,
        category: str = "processing",
        level: str = "INFO",
        function: str | None = None,
        **context: Any,
    ) -> LogEvent | None:
        """Emit one structured event to configured destinations.

        Returns ``None`` when the event is below ``min_level``; otherwise
        returns the emitted ``LogEvent`` so tests and callers can inspect it.
        """
        level_name = level.upper()
        category_name = category.lower()

        if category_name not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported log category: {category}")

        if not self.is_enabled(level_name):
            return None

        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            level=level_name,
            category=category_name,
            function=function or "unknown",
            message=message,
            elapsed_seconds=time.perf_counter() - self.started_at,
            ticker=self.ticker,
            timeframe=self.timeframe,
            context=context,
        )

        if self.log_to_console:
            print(event.to_console_line())

        if self.log_to_file:
            assert self._log_path is not None
            assert self._jsonl_path is not None
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_console_line() + "\n")
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

        return event

    def debug(self, message: str, *, function: str | None = None, **context: Any) -> LogEvent | None:
        """Emit a DEBUG diagnostic event."""
        return self.emit(message, category="debug", level="DEBUG", function=function, **context)

    def info(self, message: str, *, function: str | None = None, **context: Any) -> LogEvent | None:
        """Emit an INFO processing event."""
        return self.emit(message, category="processing", level="INFO", function=function, **context)

    def warning(self, message: str, *, function: str | None = None, **context: Any) -> LogEvent | None:
        """Emit a WARNING event."""
        return self.emit(message, category="warning", level="WARNING", function=function, **context)

    def error(self, message: str, *, function: str | None = None, **context: Any) -> LogEvent | None:
        """Emit an ERROR event."""
        return self.emit(message, category="error", level="ERROR", function=function, **context)

    def timing(self, message: str, *, function: str | None = None, **context: Any) -> LogEvent | None:
        """Emit an INFO timing event."""
        return self.emit(message, category="timing", level="INFO", function=function, **context)
