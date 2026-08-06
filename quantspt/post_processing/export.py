"""Export utilities for portfolio results.

Provides serialisation to common formats (CSV, JSON, DataFrame) for
integration with reporting pipelines and external tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require
from .discrete_allocation import AllocationResult

__all__ = [
    "to_csv",
    "to_dataframe",
    "to_json",
]


def to_dataframe(
    result: AllocationResult | NDArray[np.float64],
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Convert allocation result or weight vector to a DataFrame.

    Parameters
    ----------
    result : AllocationResult or ndarray of shape (n,)
        Either a discrete allocation result or a raw weight vector.
    tickers : list of str, optional
        Asset ticker symbols. Defaults to ``Asset_0``, ``Asset_1``, ...

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns depending on input type.
    """
    if isinstance(result, AllocationResult):
        n = len(result.shares)
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(n)]
        require(
            len(tickers) == n,
            f"tickers ({len(tickers)}) must match assets ({n})",
        )
        return pd.DataFrame(
            {
                "ticker": tickers,
                "shares": result.shares,
                "actual_weight": result.actual_weights,
            }
        )
    else:
        weights = np.asarray(result, dtype=np.float64)
        require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
        n = len(weights)
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(n)]
        require(
            len(tickers) == n,
            f"tickers ({len(tickers)}) must match assets ({n})",
        )
        return pd.DataFrame({"ticker": tickers, "weight": weights})


def to_csv(
    result: AllocationResult | NDArray[np.float64],
    path: str | Path,
    tickers: list[str] | None = None,
) -> Path:
    """Export allocation result or weights to CSV.

    Parameters
    ----------
    result : AllocationResult or ndarray of shape (n,)
        Either a discrete allocation result or a raw weight vector.
    path : str or Path
        Destination file path.
    tickers : list of str, optional
        Asset ticker symbols.

    Returns
    -------
    Path
        The path the file was written to.
    """
    path = Path(path)
    df = to_dataframe(result, tickers=tickers)
    df.to_csv(path, index=False)
    return path


def to_json(
    result: AllocationResult | NDArray[np.float64],
    path: str | Path,
    tickers: list[str] | None = None,
) -> Path:
    """Export allocation result or weights to JSON.

    Parameters
    ----------
    result : AllocationResult or ndarray of shape (n,)
        Either a discrete allocation result or a raw weight vector.
    path : str or Path
        Destination file path.
    tickers : list of str, optional
        Asset ticker symbols.

    Returns
    -------
    Path
        The path the file was written to.
    """
    path = Path(path)

    if isinstance(result, AllocationResult):
        n = len(result.shares)
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(n)]
        data: dict[str, Any] = {
            "allocations": [
                {
                    "ticker": tickers[i],
                    "shares": int(result.shares[i]),
                    "actual_weight": float(result.actual_weights[i]),
                }
                for i in range(n)
            ],
            "leftover_cash": result.leftover_cash,
        }
    else:
        weights = np.asarray(result, dtype=np.float64)
        require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
        n = len(weights)
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(n)]
        data = {
            "weights": [
                {"ticker": tickers[i], "weight": float(weights[i])} for i in range(n)
            ]
        }

    path.write_text(json.dumps(data, indent=2))
    return path
