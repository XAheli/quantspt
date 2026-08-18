"""Standard data schemas consumed by all SPT theory code.

These schemas are provider-agnostic: theory code consumes ``MarketPanel``,
``WeightVector``, and ``ReturnsMatrix`` without caring whether data came
from a CSV, Parquet, or live feed.

Mathematical References
-----------------------
- Market weights: F&K Survey Eq. 1.2
- Returns and log-returns: standard definitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "CausalGraph",
    "FactorLoadings",
    "MarketPanel",
    "RegimeLabels",
    "ReturnsMatrix",
    "WeightVector",
]


@dataclass
class MarketPanel:
    """Standard schema for market data.

    Wraps a price panel (and optional market-cap panel) with validation
    and metadata.

    Attributes
    ----------
    prices : DataFrame of shape (T, n)
        Adjusted close prices, indexed by date.
    tickers : list of str
        Asset identifiers (columns of *prices*).
    market_caps : DataFrame, optional
        Market capitalizations matching *prices* shape.
    frequency : str
        ``'daily'``, ``'weekly'``, or ``'monthly'``.
    currency : str
        Currency code (default ``'USD'``).
    """

    prices: pd.DataFrame
    tickers: list[str]
    market_caps: pd.DataFrame | None = None
    frequency: str = "daily"
    currency: str = "USD"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            not self.prices.empty,
            "Prices DataFrame must not be empty",
        )
        require(
            list(self.prices.columns) == self.tickers,
            f"Tickers {self.tickers} must match price columns {list(self.prices.columns)}",
        )
        require(
            self.frequency in ("daily", "weekly", "monthly"),
            f"Unsupported frequency: {self.frequency}",
        )
        if self.prices.isnull().all().any():
            raise ValueError("At least one ticker has all-NaN prices")

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.tickers)

    @property
    def n_observations(self) -> int:
        """Number of time observations."""
        return len(self.prices)

    @property
    def date_range(self) -> tuple[Any, Any]:
        """First and last dates in the panel."""
        return self.prices.index[0], self.prices.index[-1]

    def to_weight_vectors(self) -> pd.DataFrame:
        r"""Compute market-cap weights at each time step.

        If market_caps is available, uses those; otherwise uses
        prices as a proxy.

        .. math::
            \mu_i(t) = \frac{X_i(t)}{\sum_j X_j(t)}

        References
        ----------
        F&K Survey Eq. 1.2
        """
        caps = self.market_caps if self.market_caps is not None else self.prices
        row_sums = caps.sum(axis=1)
        return caps.div(row_sums, axis=0)


@dataclass
class WeightVector:
    """Validated portfolio or market weight vector.

    Attributes
    ----------
    weights : ndarray of shape (n,)
        Non-negative weights summing to 1.
    tickers : list of str
        Asset identifiers.
    timestamp : datetime, optional
        Point-in-time stamp.
    """

    weights: NDArray[np.float64]
    tickers: list[str]
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        require(
            self.weights.ndim == 1,
            f"Weights must be 1-D, got shape {self.weights.shape}",
        )
        require(
            len(self.weights) == len(self.tickers),
            f"Length mismatch: {len(self.weights)} weights, "
            f"{len(self.tickers)} tickers",
        )
        require(
            bool(np.all(self.weights >= -1e-10)),
            f"Weights must be non-negative, min={float(np.min(self.weights)):.2e}",
        )
        total = float(np.sum(self.weights))
        require(
            abs(total - 1.0) < 1e-6,
            f"Weights must sum to 1, got {total:.8f}",
        )


@dataclass
class ReturnsMatrix:
    """Validated returns matrix.

    Attributes
    ----------
    returns : ndarray of shape (T, n) or DataFrame
        Simple or log returns.
    tickers : list of str
        Asset identifiers.
    return_type : str
        ``'simple'`` or ``'log'``.
    """

    returns: NDArray[np.float64] | pd.DataFrame
    tickers: list[str]
    return_type: str = "simple"

    def __post_init__(self) -> None:
        shape = self.returns.shape
        require(
            len(shape) == 2,
            f"Returns must be 2-D, got shape {shape}",
        )
        n_cols = shape[1]
        require(
            n_cols == len(self.tickers),
            f"Column count {n_cols} != ticker count {len(self.tickers)}",
        )
        require(
            self.return_type in ("simple", "log"),
            f"return_type must be 'simple' or 'log', got '{self.return_type}'",
        )

    def to_numpy(self) -> NDArray[np.float64]:
        """Return the underlying array."""
        if isinstance(self.returns, pd.DataFrame):
            return self.returns.to_numpy(dtype=np.float64)
        return np.asarray(self.returns, dtype=np.float64)


@dataclass
class CausalGraph:
    """Directed graph discovered by causal structure learning.

    Attributes
    ----------
    adjacency_matrix : ndarray of shape (n, n)
        Binary or weighted adjacency matrix. Entry (i, j) > 0 indicates
        a directed edge from variable i to variable j.
    variable_names : list of str
        Names of the n variables.
    edge_weights : ndarray of shape (n, n), optional
        Edge strengths (defaults to *adjacency_matrix*).
    discovery_method : str
        Algorithm used: ``'pc'``, ``'ges'``, ``'hillclimb'``, etc.
    """

    adjacency_matrix: NDArray[np.float64]
    variable_names: list[str]
    edge_weights: NDArray[np.float64] | None = None
    discovery_method: str = "unknown"

    def __post_init__(self) -> None:
        n = len(self.variable_names)
        require(
            self.adjacency_matrix.shape == (n, n),
            f"Adjacency matrix shape {self.adjacency_matrix.shape} "
            f"does not match {n} variables",
        )


@dataclass
class RegimeLabels:
    """Regime classification output from a regime detector.

    Attributes
    ----------
    labels : ndarray of shape (T,)
        Integer regime label for each time step.
    n_regimes : int
        Number of distinct regimes.
    transition_matrix : ndarray of shape (K, K), optional
        Estimated regime transition probabilities.
    regime_statistics : dict, optional
        Per-regime summary statistics (mean, volatility, etc.).
    """

    labels: NDArray[np.int64]
    n_regimes: int
    transition_matrix: NDArray[np.float64] | None = None
    regime_statistics: dict[int, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        unique = np.unique(self.labels)
        require(
            len(unique) <= self.n_regimes,
            f"Found {len(unique)} unique labels but n_regimes={self.n_regimes}",
        )


@dataclass
class FactorLoadings:
    """Factor loading matrix from a factor model.

    Attributes
    ----------
    loadings : ndarray of shape (n, k)
        Loading matrix B mapping k factors to n assets.
    factor_names : list of str, optional
        Factor identifiers.
    explained_variance : ndarray of shape (k,), optional
        Variance explained by each factor.
    """

    loadings: NDArray[np.float64]
    factor_names: list[str] | None = None
    explained_variance: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        require(
            self.loadings.ndim == 2,
            f"Loadings must be 2-D, got shape {self.loadings.shape}",
        )
