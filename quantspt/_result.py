"""Universal result envelope for all quantspt operations.

Every public API returns an ``SPTResult`` wrapping the actual data with
metadata, warnings, and convenience accessors.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


@dataclass
class SPTResult(Generic[T]):
    """Container returned by every user-facing quantspt operation.

    Attributes
    ----------
    data
        The primary result (weights, paths, backtest metrics, …).
    metadata
        Provenance and diagnostics (estimation method, solver, …).
    warnings
        Non-fatal issues encountered during computation.
    computation_time_ms
        Wall-clock time of the computation in milliseconds.
    """

    data: T
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    computation_time_ms: float = 0.0

    # -- convenience accessors ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert *data* to a :class:`~pandas.DataFrame` when possible."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        if isinstance(self.data, np.ndarray):
            return pd.DataFrame(self.data)
        raise TypeError(f"Cannot convert {type(self.data).__name__} to DataFrame")

    def summary(self, verbose: bool = False) -> str:
        """Return a human-readable summary string."""
        parts = [f"SPTResult(type={type(self.data).__name__})"]
        if self.warnings:
            parts.append(f"  warnings: {self.warnings}")
        if verbose and self.metadata:
            parts.append(f"  metadata: {self.metadata}")
        parts.append(f"  computed in {self.computation_time_ms:.1f} ms")
        return "\n".join(parts)

    def to_json(self) -> str:
        """Serialise metadata and warnings to JSON (data is excluded)."""
        return json.dumps(
            {
                "metadata": self.metadata,
                "warnings": self.warnings,
                "computation_time_ms": self.computation_time_ms,
            },
            default=str,
        )

    def validate(self) -> bool:
        """Run basic postcondition checks and return ``True`` if valid."""
        if isinstance(self.data, np.ndarray):
            return bool(np.all(np.isfinite(self.data)))
        return True

    def __repr__(self) -> str:
        return (
            f"SPTResult(data_type={type(self.data).__name__}, "
            f"warnings={len(self.warnings)}, "
            f"time={self.computation_time_ms:.1f}ms)"
        )


class timed_result:
    """Context manager that populates *computation_time_ms* on an ``SPTResult``."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> timed_result:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
