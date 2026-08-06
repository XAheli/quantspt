"""Lot sizing and minimum trade filters for practical execution.

Handles exchange-specific constraints: many markets require trades in
round lots (e.g., 100 shares) and have minimum trade thresholds to
avoid excessive transaction costs on tiny rebalances.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "minimum_trade_filter",
    "round_to_lots",
]


def round_to_lots(
    shares: NDArray[np.intp],
    lot_size: int = 100,
) -> NDArray[np.intp]:
    """Round share counts to the nearest exchange lot size.

    Rounds each share count down to the nearest multiple of ``lot_size``
    to comply with exchange trading rules.

    Parameters
    ----------
    shares : ndarray of shape (n,)
        Integer share counts.
    lot_size : int
        Exchange lot size (default 100).

    Returns
    -------
    ndarray of shape (n,)
        Share counts rounded to multiples of ``lot_size``.
    """
    require(shares.ndim == 1, f"shares must be 1-D, got ndim={shares.ndim}")
    require(lot_size >= 1, f"lot_size must be >= 1, got {lot_size}")

    return (shares // lot_size) * lot_size


def minimum_trade_filter(
    current_shares: NDArray[np.intp],
    target_shares: NDArray[np.intp],
    min_trade: int = 1,
) -> NDArray[np.intp]:
    """Filter out trades below a minimum threshold.

    Compares current and target holdings; where the absolute difference
    is below ``min_trade``, the target is set to the current holding
    (i.e., no trade occurs).

    Parameters
    ----------
    current_shares : ndarray of shape (n,)
        Current integer share counts.
    target_shares : ndarray of shape (n,)
        Desired integer share counts after rebalancing.
    min_trade : int
        Minimum number of shares for a trade to execute (default 1).

    Returns
    -------
    ndarray of shape (n,)
        Filtered target share counts with sub-threshold trades removed.
    """
    require(
        current_shares.ndim == 1,
        f"current_shares must be 1-D, got ndim={current_shares.ndim}",
    )
    require(
        target_shares.ndim == 1,
        f"target_shares must be 1-D, got ndim={target_shares.ndim}",
    )
    require(
        len(current_shares) == len(target_shares),
        f"current_shares ({len(current_shares)}) and target_shares "
        f"({len(target_shares)}) must have same length",
    )
    require(min_trade >= 1, f"min_trade must be >= 1, got {min_trade}")

    diff = np.abs(target_shares - current_shares)
    mask = diff >= min_trade
    return np.where(mask, target_shares, current_shares)
