"""Data layer with TET (Transform-Extract-Transform) architecture.

Provides standard schemas that theory code consumes, plus pluggable
data providers for different market data sources.

Submodules
----------
schemas
    Standard data schemas: ``MarketPanel``, ``RankStats``, ``CovarianceSnapshot``.
types
    ``MarketDataFrame``, ``UniverseSpec``, and related types.
providers
    Pluggable data providers (yfinance, WRDS, Bloomberg, CSV/Parquet).
universe
    Universe construction and reconstitution logic.
corporate_actions
    Splits, dividends, M&A, and delisting adjustments.
preprocessing
    Returns computation, outlier handling, and data cleaning.
cache
    Lazy evaluation with dirty-flag caching for expensive computations.
"""

__all__: list[str] = []
