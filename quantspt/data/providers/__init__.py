"""Pluggable data providers following the TET pattern.

Each provider implements ``transform_query() -> extract_data() -> transform_data()``
and is discoverable via entry points or runtime registration.
"""

from .base import DataProvider, QueryParams
from .csv_parquet import CSVProvider, ParquetProvider
from .yfinance import YFinanceProvider

__all__ = [
    "CSVProvider",
    "DataProvider",
    "ParquetProvider",
    "QueryParams",
    "YFinanceProvider",
]
