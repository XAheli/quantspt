"""Abstract base for data providers (TET pattern).

Every data provider implements the three-step TET flow:
1. ``transform_query``: validate and normalise query parameters
2. ``extract_data``: fetch raw data from the source
3. ``transform_data``: convert raw data to a standard ``MarketPanel``

This separation ensures queries are validated before hitting the
data source, raw extraction is isolated for testing, and output
always conforms to a standard schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..._result import SPTResult
from ..schemas import MarketPanel

__all__ = [
    "DataProvider",
    "QueryParams",
]


@dataclass
class QueryParams:
    """Validated query parameters for data providers.

    Attributes
    ----------
    tickers : list of str
        Asset identifiers to fetch.
    start : datetime
        Start date for the data range.
    end : datetime
        End date for the data range.
    frequency : str
        Data frequency (``'daily'``, ``'weekly'``, ``'monthly'``).
    adjust : bool
        Whether to use adjusted prices.
    """

    tickers: list[str]
    start: datetime
    end: datetime
    frequency: str = "daily"
    adjust: bool = True


class DataProvider(ABC):
    """Abstract base class for market data providers.

    Subclasses implement the TET (Transform-Extract-Transform) flow
    for their specific data source.
    """

    @abstractmethod
    def transform_query(self, **kwargs: Any) -> QueryParams:
        """Validate and normalise query parameters.

        Parameters
        ----------
        **kwargs
            Provider-specific query arguments.

        Returns
        -------
        QueryParams
            Validated query.
        """
        ...

    @abstractmethod
    def extract_data(self, query: QueryParams) -> Any:
        """Fetch raw data from the source.

        Parameters
        ----------
        query : QueryParams
            Validated query from ``transform_query``.

        Returns
        -------
        Any
            Raw data in the provider's native format.
        """
        ...

    @abstractmethod
    def transform_data(self, raw: Any, query: QueryParams) -> MarketPanel:
        """Transform raw data into a standard ``MarketPanel``.

        Parameters
        ----------
        raw
            Raw data from ``extract_data``.
        query : QueryParams
            Original query parameters.

        Returns
        -------
        MarketPanel
            Validated market data panel.
        """
        ...

    def load(self, **kwargs: Any) -> SPTResult[MarketPanel]:
        """Run the full TET pipeline.

        Convenience method that chains ``transform_query``,
        ``extract_data``, and ``transform_data``.

        Returns
        -------
        SPTResult[MarketPanel]
            Result envelope with market data, metadata, and warnings.
        """
        from ..._result import timed_result

        with timed_result() as timer:
            query = self.transform_query(**kwargs)
            raw = self.extract_data(query)
            panel = self.transform_data(raw, query)

        return SPTResult(
            data=panel,
            metadata={
                "provider": type(self).__name__,
                "tickers": query.tickers,
                "start": str(query.start),
                "end": str(query.end),
            },
            computation_time_ms=timer.elapsed_ms,
        )
