"""Base protocol for direct optimization strategies.

Direct optimization strategies compute portfolio weights by targeting a
portfolio-level objective (e.g. excess growth rate) without committing to
a generating function. This avoids the boundary term that plagues FGPs
in concentrating markets.

This is categorically different from GeneratingFunction-based portfolios:
- No G(μ) → no boundary term log(G(μ_T)/G(μ_0))
- No structural bet on market concentration
- Weights are computed directly from the gradient of the objective
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Strategy", "WeightFunction"]

WeightFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Strategy(ABC):
    """Protocol for direct optimization portfolio strategies.

    Unlike GeneratingFunction subclasses (which use the Fernholz formula
    to derive weights from a concave function G on the simplex), Strategy
    subclasses compute weights directly from an objective and its gradient.

    The key distinction: no boundary term exists for these strategies.
    Performance is determined entirely by the rebalancing premium captured.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def compute_weights(
        self,
        market_weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute target portfolio weights.

        Parameters
        ----------
        market_weights : ndarray of shape (n,)
            Current market-capitalization weights (sum to 1, positive).
        covariance : ndarray of shape (n, n)
            Estimated covariance rate matrix (annualized).

        Returns
        -------
        ndarray of shape (n,)
            Target portfolio weights (sum to 1, non-negative).
        """
        ...

    def weight_function(self, covariance: NDArray[np.float64]) -> WeightFunction:
        """Return a weight function compatible with BacktestEngine.

        Parameters
        ----------
        covariance : ndarray of shape (n, n)
            Fixed covariance matrix to use for weight computation.

        Returns
        -------
        callable
            Function mapping market_weights → portfolio_weights.
        """

        def _wf(mu: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.compute_weights(mu, covariance)

        return _wf
