"""Trade execution simulation for backtesting.

Models the gap between desired and realized portfolio weights due to
transaction costs and market impact.

Mathematical References
-----------------------
- Proportional cost: Cost = c · Σ |Δw_i| · V (F&K Survey §9.3)
- Square-root impact: Almgren & Chriss (2000), Cost_i = η σ_i √(|Δw_i| V / ADV_i)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "ExecutionModel",
    "ExecutionResult",
    "InstantExecution",
    "MarketImpactExecution",
    "ProportionalCostExecution",
]


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a trade.

    Attributes
    ----------
    weights : ndarray of shape (n,)
        Realized portfolio weights after execution.
    cost : float
        Transaction cost as a fraction of portfolio value.
    """

    weights: NDArray[np.float64]
    cost: float


@runtime_checkable
class ExecutionModel(Protocol):
    """Protocol for trade execution simulation.

    An execution model transforms desired weight changes into realized
    weights, accounting for transaction costs and market impact.
    """

    def execute(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
    ) -> ExecutionResult:
        """Execute trades to move from current to target weights.

        Parameters
        ----------
        current_weights : ndarray of shape (n,)
            Current portfolio weights (sum to 1).
        target_weights : ndarray of shape (n,)
            Desired portfolio weights (sum to 1).
        portfolio_value : float
            Current total portfolio value.

        Returns
        -------
        ExecutionResult
            Realized weights and transaction cost fraction.
        """
        ...


@dataclass(frozen=True)
class InstantExecution:
    """Instantaneous execution at current price (zero cost).

    Baseline model: all trades execute instantly with no cost.
    """

    def execute(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
    ) -> ExecutionResult:
        """Execute instantly with zero cost."""
        return ExecutionResult(weights=target_weights.copy(), cost=0.0)


@dataclass(frozen=True)
class ProportionalCostExecution:
    r"""Proportional transaction cost model.

    Cost is proportional to turnover:

    .. math::
        \text{cost} = \frac{c}{10000} \cdot \sum_i |w_i^{\text{target}}
                      - w_i^{\text{current}}|

    where c is the cost in basis points.

    After paying costs, realized weights are the target weights applied
    to the remaining (post-cost) portfolio value. The cost reduces
    the portfolio value but the weight ratios are preserved.

    Parameters
    ----------
    cost_bps : float
        Transaction cost in basis points. Must be non-negative.
    """

    cost_bps: float

    def __post_init__(self) -> None:
        require(
            self.cost_bps >= 0.0,
            f"cost_bps must be non-negative, got {self.cost_bps}",
        )

    def execute(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
    ) -> ExecutionResult:
        """Execute with proportional transaction costs."""
        turnover = float(np.sum(np.abs(target_weights - current_weights)))
        cost_fraction = self.cost_bps / 10_000.0 * turnover
        return ExecutionResult(weights=target_weights.copy(), cost=cost_fraction)


@dataclass(frozen=True)
class MarketImpactExecution:
    r"""Square-root market impact model (Almgren-Chriss style).

    Impact cost for each asset is:

    .. math::
        \text{cost}_i = \eta \cdot \sigma_i \cdot
            \sqrt{\frac{|\Delta w_i| \cdot V}{\text{ADV}_i}}

    Total cost fraction:

    .. math::
        \text{cost} = \frac{1}{V} \sum_i \text{cost}_i \cdot |\Delta w_i| \cdot V
                    = \sum_i \eta \cdot \sigma_i \cdot |\Delta w_i|
                      \cdot \sqrt{\frac{|\Delta w_i| \cdot V}{\text{ADV}_i}}

    Parameters
    ----------
    eta : float
        Market impact coefficient. Must be positive.
    volatilities : ndarray of shape (n,)
        Annualized volatility for each asset.
    adv : ndarray of shape (n,)
        Average daily volume (in currency) for each asset.
    """

    eta: float
    volatilities: NDArray[np.float64]
    adv: NDArray[np.float64]

    def __post_init__(self) -> None:
        require(self.eta > 0.0, f"eta must be positive, got {self.eta}")
        require(
            len(self.volatilities) == len(self.adv),
            "volatilities and adv must have same length",
        )

    def execute(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
    ) -> ExecutionResult:
        """Execute with square-root market impact costs."""
        require(portfolio_value > 0.0, "portfolio_value must be positive")
        delta_w = np.abs(target_weights - current_weights)
        trade_values = delta_w * portfolio_value

        impact_per_asset = (
            self.eta * self.volatilities * np.sqrt(trade_values / self.adv)
        )
        cost_fraction = float(np.sum(impact_per_asset * delta_w))
        return ExecutionResult(weights=target_weights.copy(), cost=cost_fraction)
