"""Universe construction and time-varying asset membership.

Defines which assets are in the tradable universe at each point in time,
handling entries, exits, and filtering based on configurable criteria like
minimum observations and minimum market capitalization.

The Universe class integrates with MarketPanel to produce time-varying
membership masks for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._preconditions import require
from .schemas import MarketPanel

__all__ = [
    "Universe",
    "reconstruct",
]


@dataclass
class Universe:
    """Time-varying tradable universe definition.

    Tracks which assets are valid members of the investment universe at
    each point in time. Assets can enter (IPO, data becomes available)
    and exit (delisting, insufficient data) dynamically.

    Parameters
    ----------
    membership : DataFrame of bool, shape (T, n)
        Boolean mask indexed by dates with asset columns.
        ``True`` means the asset is a valid universe member at that time.
    tickers : list of str
        Asset identifiers matching columns of *membership*.
    metadata : dict
        Optional metadata about how the universe was constructed.
    """

    membership: pd.DataFrame
    tickers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            not self.membership.empty,
            "Membership DataFrame must not be empty",
        )
        require(
            list(self.membership.columns) == self.tickers,
            "Tickers must match membership columns",
        )
        require(
            bool(self.membership.dtypes.apply(lambda d: d == np.dtype(bool)).all()),
            "Membership must be a boolean DataFrame",
        )

    @property
    def n_dates(self) -> int:
        """Number of time observations."""
        return len(self.membership)

    @property
    def n_assets(self) -> int:
        """Total number of assets tracked."""
        return len(self.tickers)

    def members_at(self, date: Any) -> list[str]:
        """Return list of assets in the universe at a given date.

        Parameters
        ----------
        date : date-like
            Must be present in the membership index.

        Returns
        -------
        list of str
            Tickers that are universe members at that date.
        """
        row = self.membership.loc[date]
        return [t for t, v in row.items() if v]

    def member_count(self) -> pd.Series:
        """Number of active universe members at each date.

        Returns
        -------
        Series indexed by date
            Integer count of active members.
        """
        return self.membership.sum(axis=1).astype(int)

    def entry_dates(self) -> dict[str, Any]:
        """First date each asset enters the universe.

        Returns
        -------
        dict mapping ticker to first membership date (or None if never).
        """
        result: dict[str, Any] = {}
        for ticker in self.tickers:
            mask = self.membership[ticker]
            valid_dates = mask[mask].index
            result[ticker] = valid_dates[0] if len(valid_dates) > 0 else None
        return result

    def exit_dates(self) -> dict[str, Any]:
        """Last date each asset is in the universe.

        Returns
        -------
        dict mapping ticker to last membership date (or None if never a member).
        """
        result: dict[str, Any] = {}
        for ticker in self.tickers:
            mask = self.membership[ticker]
            valid_dates = mask[mask].index
            result[ticker] = valid_dates[-1] if len(valid_dates) > 0 else None
        return result

    def apply_to_panel(self, panel: MarketPanel) -> pd.DataFrame:
        """Apply universe mask to a MarketPanel, setting non-members to NaN.

        Parameters
        ----------
        panel : MarketPanel
            Must have the same date index and asset columns.

        Returns
        -------
        DataFrame
            Prices with non-member entries set to NaN.
        """
        common_dates = self.membership.index.intersection(panel.prices.index)
        common_tickers = [t for t in self.tickers if t in panel.tickers]
        require(
            len(common_dates) > 0, "No overlapping dates between universe and panel"
        )
        require(
            len(common_tickers) > 0, "No overlapping tickers between universe and panel"
        )

        prices_subset = panel.prices.loc[common_dates, common_tickers]
        mask_subset = self.membership.loc[common_dates, common_tickers]
        return prices_subset.where(mask_subset)

    def turnover(self) -> pd.DataFrame:
        """Compute entry/exit events at each date.

        Returns
        -------
        DataFrame with columns ['entries', 'exits'] indexed by date.
        """
        shifted = self.membership.shift(1)
        shifted.iloc[0] = False
        shifted = shifted.astype(bool)
        entries = self.membership & ~shifted
        exits = ~self.membership & shifted
        return pd.DataFrame(
            {"entries": entries.sum(axis=1), "exits": exits.sum(axis=1)},
            index=self.membership.index,
        )


def reconstruct(
    data: MarketPanel,
    min_observations: int = 1,
    min_market_cap: float | None = None,
    min_price: float | None = None,
    lookback_window: int | None = None,
) -> Universe:
    """Reconstruct the tradable universe from a MarketPanel.

    Applies configurable filters to determine which assets are valid
    universe members at each point in time. Membership is evaluated
    independently at each date based on historical data up to that point.

    Parameters
    ----------
    data : MarketPanel
        Market data with prices and optionally market capitalizations.
    min_observations : int
        Minimum number of non-NaN price observations required before an
        asset can enter the universe. Evaluated as a rolling count.
    min_market_cap : float, optional
        Minimum market capitalization for membership. Requires
        ``data.market_caps`` to be set.
    min_price : float, optional
        Minimum price level for membership.
    lookback_window : int, optional
        Rolling window size for evaluating observation counts.
        If ``None``, uses all available history up to each date.

    Returns
    -------
    Universe
        Time-varying membership definition.

    Examples
    --------
    >>> import pandas as pd
    >>> prices = pd.DataFrame(
    ...     {"A": [100, 101, 102, 103, 104], "B": [50, float("nan"), 52, 53, 54]},
    ...     index=pd.date_range("2020-01-01", periods=5, freq="B"),
    ... )
    >>> panel = MarketPanel(prices=prices, tickers=["A", "B"])
    >>> universe = reconstruct(panel, min_observations=2)
    >>> universe.membership.iloc[0]["A"]
    False
    """
    require(
        min_observations >= 1, f"min_observations must be >= 1, got {min_observations}"
    )

    prices = data.prices
    tickers = data.tickers
    dates = prices.index
    n_dates = len(dates)
    n_assets = len(tickers)

    membership = np.zeros((n_dates, n_assets), dtype=bool)

    not_nan = prices.notna().values

    for t in range(n_dates):
        if lookback_window is not None:
            start = max(0, t - lookback_window + 1)
        else:
            start = 0

        obs_count = not_nan[start : t + 1].sum(axis=0)
        valid = obs_count >= min_observations

        if min_price is not None:
            current_prices = np.asarray(prices.iloc[t].values, dtype=np.float64)
            price_valid = np.where(
                np.isnan(current_prices), False, current_prices >= min_price
            )
            valid = valid & price_valid

        if min_market_cap is not None and data.market_caps is not None:
            current_caps = np.asarray(data.market_caps.iloc[t].values, dtype=np.float64)
            cap_valid = np.where(
                np.isnan(current_caps), False, current_caps >= min_market_cap
            )
            valid = valid & cap_valid

        current_has_price = not_nan[t]
        valid = valid & current_has_price

        membership[t] = valid

    membership_df = pd.DataFrame(membership, index=dates, columns=tickers)

    metadata = {
        "min_observations": min_observations,
        "min_market_cap": min_market_cap,
        "min_price": min_price,
        "lookback_window": lookback_window,
    }

    return Universe(
        membership=membership_df,
        tickers=tickers,
        metadata=metadata,
    )
