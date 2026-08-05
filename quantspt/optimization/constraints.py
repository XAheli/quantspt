"""Constraint builders for SPT portfolio optimisation.

Provides composable constraint objects for use with
:func:`~quantspt.optimization.growth_rate.optimize_growth_rate`.

Supports turnover limits, position bounds, sector constraints,
and user-defined lambda constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "ConstraintSet",
    "position_limit_constraints",
    "sector_constraints",
    "turnover_constraint",
]


@dataclass
class ConstraintSet:
    """Composable container of optimisation constraints.

    Accumulates constraints and converts them to CVXPY constraint objects
    when passed to the optimiser.

    Examples
    --------
    >>> cs = ConstraintSet()
    >>> cs.add_position_limits(0.0, 0.10)
    >>> cs.add_turnover(0.20, prev_weights)
    >>> cs.add_custom(lambda w: w[0] <= 0.05)
    """

    min_weight: float | None = None
    max_weight: float | None = None
    max_turnover: float | None = None
    prev_weights: NDArray[np.float64] | None = None
    sector_map: dict[str, list[int]] | None = None
    sector_bounds: dict[str, tuple[float, float]] | None = None
    _custom_constraints: list[Callable[[NDArray[np.float64]], bool]] = field(
        default_factory=list
    )

    def add_position_limits(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> ConstraintSet:
        """Set per-asset weight bounds.

        Parameters
        ----------
        min_weight : float
            Minimum weight (0.0 for long-only).
        max_weight : float
            Maximum weight per asset.
        """
        require(min_weight <= max_weight, "min_weight must be <= max_weight")
        self.min_weight = min_weight
        self.max_weight = max_weight
        return self

    def add_turnover(
        self,
        max_turnover: float,
        prev_weights: NDArray[np.float64],
    ) -> ConstraintSet:
        """Add turnover constraint.

        Parameters
        ----------
        max_turnover : float
            Maximum one-way turnover (fraction of portfolio).
        prev_weights : ndarray of shape (n,)
            Previous portfolio weights.
        """
        require(max_turnover > 0, f"max_turnover must be positive, got {max_turnover}")
        self.max_turnover = max_turnover
        self.prev_weights = np.asarray(prev_weights, dtype=np.float64)
        return self

    def add_sector_constraints(
        self,
        sector_map: dict[str, list[int]],
        sector_bounds: dict[str, tuple[float, float]],
    ) -> ConstraintSet:
        """Add sector allocation constraints.

        Parameters
        ----------
        sector_map : dict
            Maps sector name to list of asset indices.
        sector_bounds : dict
            Maps sector name to (min_alloc, max_alloc) tuple.
        """
        self.sector_map = sector_map
        self.sector_bounds = sector_bounds
        return self

    def add_custom(
        self,
        constraint_fn: Callable[[NDArray[np.float64]], bool],
    ) -> ConstraintSet:
        """Add a user-defined constraint.

        Parameters
        ----------
        constraint_fn : callable
            Function w -> bool. Must return True when satisfied.
        """
        self._custom_constraints.append(constraint_fn)
        return self

    def to_cvxpy(self, pi: object) -> list[object]:
        """Convert to list of CVXPY constraint objects.

        Parameters
        ----------
        pi : cvxpy.Variable
            The portfolio weight variable.

        Returns
        -------
        list of cvxpy.Constraint
        """
        import cvxpy as cp

        constraints: list[object] = []

        if self.min_weight is not None:
            constraints.append(pi >= self.min_weight)  # type: ignore[operator]
        if self.max_weight is not None:
            constraints.append(pi <= self.max_weight)  # type: ignore[operator]

        if self.max_turnover is not None and self.prev_weights is not None:
            constraints.append(
                cp.norm1(pi - self.prev_weights) <= 2 * self.max_turnover  # type: ignore[operator]
            )

        if self.sector_map is not None and self.sector_bounds is not None:
            for sector, indices in self.sector_map.items():
                if sector in self.sector_bounds:
                    lo, hi = self.sector_bounds[sector]
                    sector_sum = cp.sum(pi[indices])  # type: ignore[index]
                    constraints.append(sector_sum >= lo)
                    constraints.append(sector_sum <= hi)

        return constraints

    def verify(self, weights: NDArray[np.float64]) -> bool:
        """Check whether a weight vector satisfies all constraints.

        Parameters
        ----------
        weights : ndarray of shape (n,)
            Portfolio weights to check.

        Returns
        -------
        bool
            True if all constraints are satisfied.
        """
        if self.min_weight is not None and np.any(weights < self.min_weight - 1e-8):
            return False
        if self.max_weight is not None and np.any(weights > self.max_weight + 1e-8):
            return False

        if self.max_turnover is not None and self.prev_weights is not None:
            turnover = 0.5 * float(np.sum(np.abs(weights - self.prev_weights)))
            if turnover > self.max_turnover + 1e-8:
                return False

        if self.sector_map is not None and self.sector_bounds is not None:
            for sector, indices in self.sector_map.items():
                if sector in self.sector_bounds:
                    lo, hi = self.sector_bounds[sector]
                    sec_sum = float(np.sum(weights[indices]))
                    if sec_sum < lo - 1e-8 or sec_sum > hi + 1e-8:
                        return False

        return all(fn(weights) for fn in self._custom_constraints)


def position_limit_constraints(
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> dict[str, float]:
    """Create position limit kwargs for optimize_growth_rate.

    Parameters
    ----------
    min_weight : float
        Minimum per-asset weight.
    max_weight : float
        Maximum per-asset weight.

    Returns
    -------
    dict
        kwargs to pass to optimize_growth_rate.
    """
    return {"min_weight": min_weight, "max_weight": max_weight}


def turnover_constraint(
    max_turnover: float,
    prev_weights: NDArray[np.float64],
) -> dict[str, float | NDArray[np.float64]]:
    """Create turnover kwargs for optimize_growth_rate.

    Parameters
    ----------
    max_turnover : float
        Maximum one-way turnover.
    prev_weights : ndarray
        Previous portfolio weights.

    Returns
    -------
    dict
        kwargs to pass to optimize_growth_rate.
    """
    return {
        "max_turnover": max_turnover,
        "prev_weights": np.asarray(prev_weights, dtype=np.float64),
    }


def sector_constraints(
    pi: object,
    sector_map: dict[str, list[int]],
    sector_bounds: dict[str, tuple[float, float]],
) -> list[object]:
    """Build CVXPY sector constraints.

    Parameters
    ----------
    pi : cvxpy.Variable
        Weight variable.
    sector_map : dict
        Maps sector name to list of asset indices.
    sector_bounds : dict
        Maps sector name to (min_alloc, max_alloc).

    Returns
    -------
    list of cvxpy.Constraint
    """
    import cvxpy as cp

    constraints: list[object] = []
    for sector, indices in sector_map.items():
        if sector in sector_bounds:
            lo, hi = sector_bounds[sector]
            sector_sum = cp.sum(pi[indices])  # type: ignore[index]
            constraints.append(sector_sum >= lo)
            constraints.append(sector_sum <= hi)
    return constraints
