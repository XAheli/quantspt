"""Data preprocessing: returns, missing data, outliers, and filtering.

Provides utilities for transforming raw price data into analysis-ready
returns matrices, with configurable handling of missing data, outliers,
and universe filtering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require
from .schemas import ReturnsMatrix

__all__ = [
    "compute_log_returns",
    "compute_simple_returns",
    "detect_outliers",
    "filter_universe",
    "handle_missing",
    "winsorise",
]


def compute_simple_returns(
    prices: pd.DataFrame | NDArray[np.float64],
    drop_first: bool = True,
) -> ReturnsMatrix | NDArray[np.float64]:
    r"""Compute simple (arithmetic) returns from prices.

    .. math::
        R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1

    Parameters
    ----------
    prices : DataFrame or ndarray of shape (T, n)
        Price panel.
    drop_first : bool
        Whether to drop the first row (which is NaN).

    Returns
    -------
    ReturnsMatrix (if DataFrame input) or ndarray
    """
    if isinstance(prices, pd.DataFrame):
        rets = prices.pct_change()
        if drop_first:
            rets = rets.iloc[1:]
        return ReturnsMatrix(
            returns=rets,
            tickers=list(prices.columns),
            return_type="simple",
        )

    require(prices.ndim == 2, f"Prices must be 2-D, got ndim={prices.ndim}")
    rets_arr = np.diff(prices, axis=0) / prices[:-1]
    if not drop_first:
        first_row = np.full((1, prices.shape[1]), np.nan)
        rets_arr = np.vstack([first_row, rets_arr])
    return rets_arr


def compute_log_returns(
    prices: pd.DataFrame | NDArray[np.float64],
    drop_first: bool = True,
) -> ReturnsMatrix | NDArray[np.float64]:
    r"""Compute log (continuously-compounded) returns from prices.

    .. math::
        r_t = \ln\frac{P_t}{P_{t-1}}

    Parameters
    ----------
    prices : DataFrame or ndarray of shape (T, n)
        Price panel.  Must be strictly positive.
    drop_first : bool
        Whether to drop the first row (which is NaN).

    Returns
    -------
    ReturnsMatrix (if DataFrame input) or ndarray
    """
    if isinstance(prices, pd.DataFrame):
        ratio = prices / prices.shift(1)
        log_rets_df = ratio.apply(np.log)
        if drop_first:
            log_rets_df = log_rets_df.iloc[1:]
        return ReturnsMatrix(
            returns=log_rets_df,
            tickers=list(prices.columns),
            return_type="log",
        )

    require(prices.ndim == 2, f"Prices must be 2-D, got ndim={prices.ndim}")
    require(bool(np.all(prices > 0)), "Prices must be positive for log returns")
    log_rets_arr = np.diff(np.log(prices), axis=0)
    if not drop_first:
        first_row = np.full((1, prices.shape[1]), np.nan)
        log_rets_arr = np.vstack([first_row, log_rets_arr])
    return log_rets_arr


def handle_missing(
    data: pd.DataFrame,
    method: str = "ffill",
    max_gap: int | None = None,
) -> pd.DataFrame:
    """Handle missing data in a price or returns panel.

    Parameters
    ----------
    data : DataFrame
        Data with potential NaN values.
    method : str
        ``'ffill'`` (forward-fill), ``'drop'`` (drop rows with any NaN),
        or ``'interpolate'`` (linear interpolation).
    max_gap : int, optional
        Maximum consecutive NaN gap to fill (for ``'ffill'`` and
        ``'interpolate'``).  Larger gaps remain NaN.

    Returns
    -------
    DataFrame
        Cleaned data.
    """
    require(
        method in ("ffill", "drop", "interpolate"),
        f"Unknown method: {method}. Use 'ffill', 'drop', or 'interpolate'.",
    )

    if method == "ffill":
        return data.ffill(limit=max_gap)
    elif method == "drop":
        return data.dropna()
    else:
        return data.interpolate(method="linear", limit=max_gap)


def detect_outliers(
    returns: pd.DataFrame | NDArray[np.float64],
    method: str = "zscore",
    threshold: float = 3.0,
) -> NDArray[np.bool_]:
    """Detect outliers in a returns matrix.

    Parameters
    ----------
    returns : DataFrame or ndarray of shape (T, n)
        Returns data.
    method : str
        ``'zscore'`` (|z| > threshold) or ``'iqr'``
        (outside Q1 - k*IQR, Q3 + k*IQR where k=threshold).
    threshold : float
        Detection threshold (default 3.0 for z-score, 1.5 for IQR).

    Returns
    -------
    ndarray of bool, shape (T, n)
        ``True`` where an outlier is detected.
    """
    if isinstance(returns, pd.DataFrame):
        arr: NDArray[np.float64] = returns.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(returns, dtype=np.float64)

    require(arr.ndim == 2, f"Returns must be 2-D, got ndim={arr.ndim}")
    require(
        method in ("zscore", "iqr"),
        f"Unknown method: {method}. Use 'zscore' or 'iqr'.",
    )

    if method == "zscore":
        col_mean = np.nanmean(arr, axis=0)
        col_std = np.nanstd(arr, axis=0, ddof=1)
        col_std = np.where(col_std > 0, col_std, 1.0)
        z = np.abs((arr - col_mean) / col_std)
        result: NDArray[np.bool_] = z > threshold
        return result

    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    iqr_val = q3 - q1
    lower_bound = q1 - threshold * iqr_val
    upper_bound = q3 + threshold * iqr_val
    result = (arr < lower_bound) | (arr > upper_bound)
    return result


def winsorise(
    returns: pd.DataFrame | NDArray[np.float64],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> pd.DataFrame | NDArray[np.float64]:
    """Winsorise returns by clipping to percentile bounds.

    Parameters
    ----------
    returns : DataFrame or ndarray
        Returns data.
    lower_pct : float
        Lower percentile cutoff (default 1%).
    upper_pct : float
        Upper percentile cutoff (default 99%).

    Returns
    -------
    Same type as input, with extreme values clipped.
    """
    if isinstance(returns, pd.DataFrame):
        vals: NDArray[np.float64] = returns.to_numpy(dtype=np.float64)
        lo = np.nanpercentile(vals, lower_pct, axis=0)
        hi = np.nanpercentile(vals, upper_pct, axis=0)
        clipped = np.clip(vals, lo, hi)
        return pd.DataFrame(clipped, index=returns.index, columns=returns.columns)

    vals = np.asarray(returns, dtype=np.float64)
    lo = np.nanpercentile(vals, lower_pct, axis=0)
    hi = np.nanpercentile(vals, upper_pct, axis=0)
    return np.clip(vals, lo, hi)


def filter_universe(
    prices: pd.DataFrame,
    min_observations: int | None = None,
    min_non_nan_fraction: float = 0.8,
) -> pd.DataFrame:
    """Filter universe to assets meeting data quality criteria.

    Parameters
    ----------
    prices : DataFrame
        Price panel.
    min_observations : int, optional
        Minimum number of non-NaN observations per asset.
        If ``None``, uses ``min_non_nan_fraction`` instead.
    min_non_nan_fraction : float
        Minimum fraction of non-NaN observations (default 0.8).

    Returns
    -------
    DataFrame
        Filtered price panel with only qualifying assets.
    """
    n_obs = prices.count()
    total = len(prices)

    if min_observations is not None:
        threshold_val = min_observations
    else:
        threshold_val = int(total * min_non_nan_fraction)

    kept_cols = [col for col in prices.columns if int(n_obs[col]) >= threshold_val]
    require(len(kept_cols) > 0, "No assets pass the universe filter")
    return prices[kept_cols]
