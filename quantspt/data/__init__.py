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
universe
    Universe construction and time-varying asset membership.
corporate_actions
    Corporate action handling (splits, dividends, delistings).
"""

from .corporate_actions import (
    adjust_for_dividends,
    adjust_for_splits,
    detect_splits,
    handle_delistings,
)
from .schemas import MarketPanel, ReturnsMatrix, WeightVector
from .universe import Universe, reconstruct

__all__ = [
    "MarketPanel",
    "ReturnsMatrix",
    "Universe",
    "WeightVector",
    "adjust_for_dividends",
    "adjust_for_splits",
    "detect_splits",
    "handle_delistings",
    "reconstruct",
]
