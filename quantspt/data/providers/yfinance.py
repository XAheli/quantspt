"""Yahoo Finance data provider.

Fetches price data from Yahoo Finance via the ``yfinance`` library.
Returns data in the standard ``MarketPanel`` schema for immediate use
with quantspt's estimation, backtesting, and visualization modules.

Requires the ``data`` extra: ``pip install quantspt[data]``
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ..._preconditions import require
from ...errors import DataProviderError
from ..schemas import MarketPanel
from .base import DataProvider, QueryParams

__all__ = ["YFinanceProvider"]


def _require_yfinance() -> Any:
    """Import yfinance or raise with installation instructions."""
    try:
        import yfinance

        return yfinance
    except ImportError as exc:
        raise ImportError(
            "YFinanceProvider requires yfinance. "
            "Install with: pip install quantspt[data]"
        ) from exc


class YFinanceProvider(DataProvider):
    """Fetch market data from Yahoo Finance.

    Wraps ``yfinance.download()`` behind the standard TET flow
    (Transform-Extract-Transform), providing validated ``MarketPanel``
    output suitable for all quantspt modules.

    Parameters
    ----------
    auto_adjust : bool
        Use adjusted close prices (accounting for dividends and splits).
        Default True.
    progress : bool
        Show the yfinance download progress bar. Default False.

    Examples
    --------
    >>> from quantspt.data.providers.yfinance import YFinanceProvider
    >>> provider = YFinanceProvider()
    >>> result = provider.load(
    ...     tickers=["AAPL", "MSFT", "GOOG"],
    ...     start="2020-01-01",
    ...     end="2023-12-31",
    ... )
    >>> panel = result.data
    >>> print(panel.prices.head())
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> None:
        _require_yfinance()
        self._auto_adjust = auto_adjust
        self._progress = progress

    def transform_query(self, **kwargs: Any) -> QueryParams:
        """Validate and normalise query parameters.

        Parameters
        ----------
        **kwargs
            ``tickers`` : list of str — ticker symbols (required).
            ``start`` : str or datetime — start date (default 5 years ago).
            ``end`` : str or datetime — end date (default today).
            ``frequency`` : str — 'daily', 'weekly', 'monthly' (default 'daily').
            ``adjust`` : bool — use adjusted prices (default True).

        Returns
        -------
        QueryParams
        """
        tickers = kwargs.get("tickers", [])
        require(len(tickers) > 0, "At least one ticker is required.")

        start = kwargs.get("start", datetime(datetime.now().year - 5, 1, 1))
        end = kwargs.get("end", datetime.now())
        frequency = kwargs.get("frequency", "daily")
        adjust = kwargs.get("adjust", self._auto_adjust)

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        return QueryParams(
            tickers=tickers,
            start=start,
            end=end,
            frequency=frequency,
            adjust=adjust,
        )

    def extract_data(self, query: QueryParams) -> pd.DataFrame:
        """Download raw price data from Yahoo Finance.

        Parameters
        ----------
        query : QueryParams
            Validated query from ``transform_query``.

        Returns
        -------
        pd.DataFrame
            Raw OHLCV data indexed by date.

        Raises
        ------
        DataProviderError
            On network errors, invalid tickers, or empty responses.
        """
        yf = _require_yfinance()

        interval_map = {
            "daily": "1d",
            "weekly": "1wk",
            "monthly": "1mo",
        }
        interval = interval_map.get(query.frequency, "1d")

        try:
            data = yf.download(
                tickers=query.tickers,
                start=query.start.strftime("%Y-%m-%d"),
                end=query.end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=query.adjust,
                progress=self._progress,
            )
        except Exception as exc:
            raise DataProviderError(
                f"Failed to download data from Yahoo Finance: {exc}"
            ) from exc

        if data is None or data.empty:
            raise DataProviderError(
                f"No data returned for tickers {query.tickers} "
                f"between {query.start} and {query.end}."
            )

        return data

    def transform_data(self, raw: pd.DataFrame, query: QueryParams) -> MarketPanel:
        """Transform raw yfinance data into a ``MarketPanel``.

        Extracts the "Close" (or "Adj Close") price series, handles
        multi-ticker column hierarchies, and validates the result.

        Parameters
        ----------
        raw : pd.DataFrame
            Raw data from ``extract_data``.
        query : QueryParams
            Original query parameters.

        Returns
        -------
        MarketPanel
        """
        prices_df: pd.DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                extracted = raw["Close"]
            elif "Adj Close" in raw.columns.get_level_values(0):
                extracted = raw["Adj Close"]
            else:
                subset = raw.iloc[
                    :,
                    raw.columns.get_level_values(0)
                    == raw.columns.get_level_values(0)[0],
                ]
                subset.columns = subset.columns.droplevel(0)
                extracted = subset

            if isinstance(extracted, pd.Series):
                prices_df = extracted.to_frame(name=query.tickers[0])
            else:
                prices_df = extracted
        else:
            prices_df = raw

        for ticker in query.tickers:
            if ticker not in prices_df.columns:
                raise DataProviderError(
                    f"Ticker '{ticker}' not found in downloaded data. "
                    f"Available columns: {list(prices_df.columns)}"
                )

        prices_df = prices_df[query.tickers]
        prices_df = prices_df.apply(pd.to_numeric, errors="coerce")

        n_before = len(prices_df)
        prices_df = prices_df.dropna(how="all")
        n_dropped = n_before - len(prices_df)

        prices_df = prices_df.ffill()

        remaining_nans = int(prices_df.isna().sum().sum())
        if remaining_nans > 0:
            prices_df = prices_df.dropna(how="any")

        if prices_df.empty or len(prices_df) == 0:
            raise DataProviderError(
                "Data contains unresolvable NaN values after forward-fill "
                "(no look-ahead bias: bfill is not used)."
            )

        remaining_nans = int(prices_df.isna().sum().sum())
        if remaining_nans > 0:
            raise DataProviderError(
                f"Data contains {remaining_nans} NaN values after forward-fill."
            )

        tickers = list(prices_df.columns)

        panel = MarketPanel(
            prices=prices_df.astype(float),
            tickers=tickers,
        )

        if n_dropped > 0:
            panel.metadata["rows_dropped_all_nan"] = n_dropped

        return panel
