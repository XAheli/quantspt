"""Rebalancing triggers for the backtesting engine.

Determines when a portfolio should rebalance. Each trigger implements the
``Rebalancer`` protocol and can be composed or swapped independently.

Mathematical References
-----------------------
- Portfolio drift from target: max_i |π_i(t) - π*_i| or L2 norm
- Calendar schedules: standard trading calendar periods
- Threshold triggers: band-based rebalancing (Leland 2006 style)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "CalendarRebalancer",
    "DriftRebalancer",
    "Frequency",
    "Rebalancer",
    "ThresholdRebalancer",
]


@unique
class Frequency(Enum):
    """Standard rebalancing frequencies."""

    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63


@runtime_checkable
class Rebalancer(Protocol):
    """Protocol for rebalancing triggers.

    A rebalancer decides whether the portfolio should rebalance at a given
    time step, based on the current and target weights.
    """

    def should_rebalance(
        self,
        step: int,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
    ) -> bool:
        """Return True if the portfolio should rebalance at this step.

        Parameters
        ----------
        step : int
            Current time step index (0-based).
        current_weights : ndarray of shape (n,)
            Weights after market movement (before any rebalance).
        target_weights : ndarray of shape (n,)
            Desired portfolio weights from the strategy.

        Returns
        -------
        bool
            True if rebalancing should occur.
        """
        ...


@dataclass(frozen=True)
class CalendarRebalancer:
    """Rebalance on a fixed calendar schedule.

    Triggers rebalancing every ``frequency`` trading days. The first
    rebalance always occurs at step 0.

    Parameters
    ----------
    frequency : Frequency
        How often to rebalance (daily, weekly, monthly, quarterly).
    """

    frequency: Frequency

    def should_rebalance(
        self,
        step: int,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
    ) -> bool:
        """Rebalance every ``frequency`` steps, always at step 0."""
        return step % self.frequency.value == 0


@dataclass(frozen=True)
class ThresholdRebalancer:
    r"""Rebalance when max absolute weight drift exceeds a threshold.

    Triggers when:

    .. math::
        \max_i |w_i^{\text{current}} - w_i^{\text{target}}| > \text{threshold}

    Always rebalances at step 0 to establish initial positions.

    Parameters
    ----------
    threshold : float
        Maximum allowed drift per asset before rebalancing.
        Must be in (0, 1).
    """

    threshold: float

    def __post_init__(self) -> None:
        require(
            0.0 < self.threshold < 1.0,
            f"Threshold must be in (0, 1), got {self.threshold}",
        )

    def should_rebalance(
        self,
        step: int,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
    ) -> bool:
        """Rebalance when max weight drift exceeds threshold."""
        if step == 0:
            return True
        max_drift = float(np.max(np.abs(current_weights - target_weights)))
        return max_drift > self.threshold


@dataclass(frozen=True)
class DriftRebalancer:
    r"""Rebalance when total portfolio drift from target exceeds a limit.

    Uses L2 norm of weight differences:

    .. math::
        \|w^{\text{current}} - w^{\text{target}}\|_2 > \text{max\_drift}

    Always rebalances at step 0.

    Parameters
    ----------
    max_drift : float
        Maximum allowed L2 drift before rebalancing. Must be positive.
    """

    max_drift: float

    def __post_init__(self) -> None:
        require(
            self.max_drift > 0.0,
            f"max_drift must be positive, got {self.max_drift}",
        )

    def should_rebalance(
        self,
        step: int,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
    ) -> bool:
        """Rebalance when L2 drift exceeds max_drift."""
        if step == 0:
            return True
        l2_drift = float(np.linalg.norm(current_weights - target_weights))
        return l2_drift > self.max_drift
