"""CSV and Parquet data providers.

Load market data from local CSV or Parquet files, with automatic
date column detection and missing-data handling.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..._preconditions import require
from ...errors import DataProviderError
from ..schemas import MarketPanel
from .base import DataProvider, QueryParams

__all__ = [
    "CSVProvider",
    "ParquetProvider",
]


_DATE_COLUMNS = {"date", "Date", "DATE", "timestamp", "Timestamp", "time", "Time"}


def _detect_date_column(df: pd.DataFrame) -> str | None:
    """Try to find a date column by name or dtype."""
    for col in df.columns:
        if col in _DATE_COLUMNS:
            return str(col)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    return None


def _parse_dates(df: pd.DataFrame, date_column: str | None) -> pd.DataFrame:
    """Set index to a datetime column, detecting if needed."""
    if date_column is None:
        date_column = _detect_date_column(df)

    if date_column is not None and date_column in df.columns:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError) as exc:
            raise DataProviderError(f"Could not parse dates from index: {exc}") from exc

    df = df.sort_index()
    return df


class CSVProvider(DataProvider):
    """Load market data from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    date_column : str, optional
        Name of the date column.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        path: str | Path,
        date_column: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._date_column = date_column

    def transform_query(self, **kwargs: Any) -> QueryParams:
        tickers = kwargs.get("tickers", [])
        start = kwargs.get("start", datetime(1900, 1, 1))
        end = kwargs.get("end", datetime(2100, 1, 1))
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        return QueryParams(tickers=tickers, start=start, end=end)

    def extract_data(self, query: QueryParams) -> pd.DataFrame:
        require(self._path.exists(), f"File not found: {self._path}")
        df = pd.read_csv(self._path)
        return _parse_dates(df, self._date_column)

    def transform_data(self, raw: pd.DataFrame, query: QueryParams) -> MarketPanel:
        df = raw

        if query.tickers:
            missing = [t for t in query.tickers if t not in df.columns]
            if missing:
                raise DataProviderError(f"Tickers not found in data: {missing}")
            df = df[query.tickers]

        if isinstance(df.index, pd.DatetimeIndex):
            mask = (df.index >= pd.Timestamp(query.start)) & (
                df.index <= pd.Timestamp(query.end)
            )
            df = df.loc[mask]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.columns)
        df = df[numeric_cols]

        tickers = list(df.columns)
        require(len(tickers) > 0, "No asset columns found in data")

        df = df.ffill()

        return MarketPanel(
            prices=df.astype(float),
            tickers=tickers,
        )


class ParquetProvider(DataProvider):
    """Load market data from a Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the Parquet file.
    date_column : str, optional
        Name of the date column.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        path: str | Path,
        date_column: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._date_column = date_column

    def transform_query(self, **kwargs: Any) -> QueryParams:
        tickers = kwargs.get("tickers", [])
        start = kwargs.get("start", datetime(1900, 1, 1))
        end = kwargs.get("end", datetime(2100, 1, 1))
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        return QueryParams(tickers=tickers, start=start, end=end)

    def extract_data(self, query: QueryParams) -> pd.DataFrame:
        require(self._path.exists(), f"File not found: {self._path}")
        df = pd.read_parquet(self._path)
        return _parse_dates(df, self._date_column)

    def transform_data(self, raw: pd.DataFrame, query: QueryParams) -> MarketPanel:
        df = raw

        if query.tickers:
            missing = [t for t in query.tickers if t not in df.columns]
            if missing:
                raise DataProviderError(f"Tickers not found in data: {missing}")
            df = df[query.tickers]

        if isinstance(df.index, pd.DatetimeIndex):
            mask = (df.index >= pd.Timestamp(query.start)) & (
                df.index <= pd.Timestamp(query.end)
            )
            df = df.loc[mask]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            numeric_cols = list(df.columns)
        df = df[numeric_cols]

        tickers = list(df.columns)
        require(len(tickers) > 0, "No asset columns found in data")

        df = df.ffill()

        return MarketPanel(
            prices=df.astype(float),
            tickers=tickers,
        )
