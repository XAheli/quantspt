"""Corporate action handling for historical price adjustment.

Provides utilities for adjusting historical price data for stock splits,
dividends, and delistings. Includes automatic split detection from
anomalous price movements.

These adjustments are critical for accurate backtesting: unadjusted data
creates phantom returns at split/dividend dates that distort portfolio
performance calculations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "adjust_for_dividends",
    "adjust_for_splits",
    "detect_splits",
    "handle_delistings",
]


def adjust_for_splits(
    prices: pd.DataFrame | NDArray[np.float64],
    split_ratios: pd.DataFrame | dict[str, list[tuple[Any, float]]],
) -> pd.DataFrame | NDArray[np.float64]:
    """Adjust historical prices for stock splits.

    Applies split ratios retroactively: prices before a split date are
    divided by the cumulative split ratio to make the series continuous.

    Parameters
    ----------
    prices : DataFrame of shape (T, n) or ndarray
        Unadjusted price panel.
    split_ratios : DataFrame or dict
        If DataFrame: same shape as prices, with split ratio values at
        split dates (e.g., 2.0 for a 2:1 split) and 1.0 elsewhere.
        If dict: maps ticker → list of (date, ratio) tuples.

    Returns
    -------
    Same type as input
        Split-adjusted prices where the entire history is scaled to be
        consistent with the most recent price level.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2020-01-01", periods=4, freq="B")
    >>> prices = pd.DataFrame({"A": [100.0, 102.0, 51.0, 52.0]}, index=dates)
    >>> splits = {"A": [(dates[2], 2.0)]}
    >>> adjusted = adjust_for_splits(prices, splits)
    >>> adjusted["A"].iloc[0]  # 100 / 2 = 50
    50.0
    """
    if isinstance(prices, np.ndarray):
        require(prices.ndim == 2, f"Prices must be 2-D, got ndim={prices.ndim}")
        if not isinstance(split_ratios, pd.DataFrame):
            raise TypeError(
                "For ndarray prices, split_ratios must be a DataFrame with same shape"
            )
        ratios_arr: NDArray[np.float64] = np.asarray(
            split_ratios.values, dtype=np.float64
        )
        cum_ratios = np.cumprod(ratios_arr[::-1], axis=0)[::-1]
        adjustment = cum_ratios / cum_ratios[-1]
        return prices / adjustment

    require(isinstance(prices, pd.DataFrame), "prices must be DataFrame or ndarray")
    assert isinstance(prices, pd.DataFrame)

    adjusted = prices.copy()

    if isinstance(split_ratios, dict):
        for ticker, splits_list in split_ratios.items():
            require(
                ticker in adjusted.columns,
                f"Ticker '{ticker}' not found in prices columns",
            )
            for split_date, ratio in sorted(splits_list, key=lambda x: x[0]):
                require(ratio > 0, f"Split ratio must be positive, got {ratio}")
                mask = adjusted.index < split_date
                adjusted.loc[mask, ticker] = adjusted.loc[mask, ticker] / ratio
    elif isinstance(split_ratios, pd.DataFrame):
        require(
            split_ratios.shape == prices.shape,
            f"split_ratios shape {split_ratios.shape} != prices shape {prices.shape}",
        )
        cum_ratios_df = split_ratios.iloc[::-1].cumprod().iloc[::-1]
        last_row = cum_ratios_df.iloc[-1]
        adjustment_df = cum_ratios_df.div(last_row, axis=1)
        adjusted = prices / adjustment_df

    return adjusted


def adjust_for_dividends(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    method: str = "total_return",
) -> pd.DataFrame:
    """Adjust prices for dividend payments.

    Parameters
    ----------
    prices : DataFrame of shape (T, n)
        Price panel (ex-dividend prices if using 'total_return').
    dividends : DataFrame of shape (T, n)
        Dividend amounts on ex-dates. Zero where no dividend.
    method : str
        - ``'total_return'``: Reinvest dividends proportionally, creating a
          total-return index. Adjusts all pre-dividend prices upward.
        - ``'price_only'``: No adjustment; returns prices unchanged.
          Use when dividends are tracked separately.
        - ``'proportional'``: Adjust pre-dividend prices by the
          dividend yield factor (1 - div/price_before_div).

    Returns
    -------
    DataFrame
        Dividend-adjusted price panel.

    Notes
    -----
    The total return method ensures that the adjusted price series
    reflects the full economic return to a buy-and-hold investor who
    reinvests all dividends at the prevailing price.
    """
    require(
        method in ("total_return", "price_only", "proportional"),
        f"Unknown method '{method}'. Use 'total_return', 'price_only', or 'proportional'.",
    )
    require(
        prices.shape == dividends.shape,
        f"Prices shape {prices.shape} != dividends shape {dividends.shape}",
    )

    if method == "price_only":
        return prices.copy()

    adjusted = prices.copy()

    if method == "total_return":
        div_dates = dividends.index[dividends.any(axis=1)]
        for div_date in reversed(list(div_dates)):
            date_idx_raw = prices.index.get_loc(div_date)
            assert isinstance(date_idx_raw, int)
            if date_idx_raw == 0:
                continue
            prev_date = prices.index[date_idx_raw - 1]
            prev_prices = adjusted.loc[prev_date]
            div_amounts = dividends.loc[div_date]
            safe_prev = prev_prices.where(prev_prices > 0, other=np.nan)
            factor = 1.0 + div_amounts / safe_prev
            factor = factor.fillna(1.0)
            mask = adjusted.index < div_date
            adjusted.loc[mask] = adjusted.loc[mask].mul(1.0 / factor, axis=1)

    elif method == "proportional":
        div_dates = dividends.index[dividends.any(axis=1)]
        for div_date in reversed(list(div_dates)):
            date_idx_raw = prices.index.get_loc(div_date)
            assert isinstance(date_idx_raw, int)
            if date_idx_raw == 0:
                continue
            prev_date = prices.index[date_idx_raw - 1]
            prev_prices = adjusted.loc[prev_date]
            div_amounts = dividends.loc[div_date]
            safe_prev = prev_prices.where(prev_prices > 0, other=np.nan)
            factor = 1.0 - div_amounts / safe_prev
            factor = factor.fillna(1.0)
            mask = adjusted.index < div_date
            adjusted.loc[mask] = adjusted.loc[mask].mul(factor, axis=1)

    return adjusted


def handle_delistings(
    prices: pd.DataFrame,
    delisted: dict[str, Any] | None = None,
    method: str = "last_price",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Handle delisted assets in a price panel.

    Parameters
    ----------
    prices : DataFrame of shape (T, n)
        Price panel where delisted stocks may have NaN tails.
    delisted : dict, optional
        Mapping of ticker → delisting date. If ``None``, delistings are
        inferred from trailing NaN sequences.
    method : str
        - ``'last_price'``: Forward-fill with the last known price.
        - ``'zero'``: Set post-delisting prices to zero (total loss).
        - ``'nan'``: Leave as NaN (exclude from computations).
        - ``'fill_value'``: Set to the specified ``fill_value``.

    fill_value : float
        Value to use when ``method='fill_value'``.

    Returns
    -------
    DataFrame
        Prices with delistings handled according to the chosen method.
    """
    require(
        method in ("last_price", "zero", "nan", "fill_value"),
        f"Unknown method '{method}'. Use 'last_price', 'zero', 'nan', or 'fill_value'.",
    )

    result = prices.copy()

    if delisted is None:
        delisted = _infer_delistings(prices)

    for ticker, delist_date in delisted.items():
        if ticker not in result.columns:
            continue
        mask = result.index > delist_date

        if method == "last_price":
            last_valid_idx = result[ticker].last_valid_index()
            if last_valid_idx is not None:
                last_price = float(result[ticker].loc[last_valid_idx])
                result.loc[mask, ticker] = last_price
        elif method == "zero":
            result.loc[mask, ticker] = 0.0
        elif method == "nan":
            result.loc[mask, ticker] = np.nan
        elif method == "fill_value":
            result.loc[mask, ticker] = fill_value

    return result


def _infer_delistings(prices: pd.DataFrame) -> dict[str, Any]:
    """Infer delisting dates from trailing NaN sequences.

    A stock is considered delisted if it has a continuous sequence of
    NaN values extending to the end of the panel.

    Returns
    -------
    dict
        Mapping ticker → last valid date (inferred delisting date).
    """
    delisted: dict[str, Any] = {}
    for ticker in prices.columns:
        col = prices[ticker]
        if col.isna().iloc[-1]:
            last_valid = col.last_valid_index()
            if last_valid is not None:
                delisted[ticker] = last_valid
    return delisted


def detect_splits(
    prices: pd.DataFrame | NDArray[np.float64],
    threshold: float = 0.4,
    common_ratios: tuple[float, ...] = (2.0, 3.0, 4.0, 0.5, 0.333, 0.25),
    ratio_tolerance: float = 0.05,
) -> dict[str, list[tuple[Any, float]]] | list[tuple[int, int, float]]:
    """Auto-detect probable stock splits from price jumps.

    Identifies dates where the price ratio between consecutive days
    is close to a common split ratio (within tolerance), suggesting
    an unadjusted split.

    Parameters
    ----------
    prices : DataFrame or ndarray of shape (T, n)
        Price panel (potentially unadjusted).
    threshold : float
        Minimum absolute log-return to flag as a potential split.
        Default 0.4 corresponds to ~50% single-day move.
    common_ratios : tuple of float
        Split ratios to test against. Forward splits (>1) and reverse
        splits (<1) are both included.
    ratio_tolerance : float
        How close the actual ratio must be to a common split ratio.

    Returns
    -------
    dict (if DataFrame) or list of tuples (if ndarray)
        For DataFrame: maps ticker → list of (date, detected_ratio).
        For ndarray: list of (time_index, asset_index, detected_ratio).
    """
    require(threshold > 0, f"threshold must be positive, got {threshold}")
    require(
        ratio_tolerance > 0, f"ratio_tolerance must be positive, got {ratio_tolerance}"
    )

    if isinstance(prices, np.ndarray):
        require(prices.ndim == 2, f"Prices must be 2-D, got ndim={prices.ndim}")
        return _detect_splits_array(prices, threshold, common_ratios, ratio_tolerance)

    results: dict[str, list[tuple[Any, float]]] = {}

    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < 2:
            continue

        col_vals: NDArray[np.float64] = np.asarray(col.values, dtype=np.float64)
        ratios = col_vals[1:] / col_vals[:-1]
        log_rets = np.log(ratios)
        large_moves = np.where(np.abs(log_rets) > threshold)[0]

        detected: list[tuple[Any, float]] = []
        for idx in large_moves:
            actual_ratio = ratios[idx]
            for split_ratio in common_ratios:
                inv_ratio = 1.0 / split_ratio
                if abs(actual_ratio - inv_ratio) < ratio_tolerance:
                    date = col.index[idx + 1]
                    detected.append((date, split_ratio))
                    break

        if detected:
            results[ticker] = detected

    return results


def _detect_splits_array(
    prices: NDArray[np.float64],
    threshold: float,
    common_ratios: tuple[float, ...],
    ratio_tolerance: float,
) -> list[tuple[int, int, float]]:
    """Detect splits in a numpy array of prices."""
    T, n = prices.shape
    results: list[tuple[int, int, float]] = []

    for j in range(n):
        col = prices[:, j]
        valid_mask = ~np.isnan(col)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            continue

        for k in range(1, len(valid_indices)):
            i_curr = valid_indices[k]
            i_prev = valid_indices[k - 1]
            ratio = col[i_curr] / col[i_prev]

            if abs(np.log(ratio)) > threshold:
                for split_ratio in common_ratios:
                    inv_ratio = 1.0 / split_ratio
                    if abs(ratio - inv_ratio) < ratio_tolerance:
                        results.append((i_curr, j, split_ratio))
                        break

    return results
