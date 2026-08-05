"""Data layer with TET (Transform-Extract-Transform) architecture.

Provides standard schemas that theory code consumes, plus pluggable
data providers for different market data sources.

Submodules
----------
schemas
    Standard data schemas: ``MarketPanel``, ``WeightVector``, ``ReturnsMatrix``.
providers
    Pluggable data providers (CSV, Parquet, and extensible base).
preprocessing
    Returns computation, outlier handling, and data cleaning.
"""

from .schemas import MarketPanel, ReturnsMatrix, WeightVector

__all__ = [
    "MarketPanel",
    "ReturnsMatrix",
    "WeightVector",
]
