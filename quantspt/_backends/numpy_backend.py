"""NumPy compute backend — the default implementation.

Provides explicit implementations of core SPT operations using NumPy.
This is the reference backend that other backends must match in behavior.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["NumpyBackend"]


class NumpyBackend:
    """NumPy-based reference implementation of core SPT operations."""

    @property
    def name(self) -> str:
        return "numpy"

    def excess_growth_rate(
        self,
        pi: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> float:
        r"""Compute excess growth rate γ*_π = ½(Σ π_i a_{ii} - π'aπ)."""
        weighted_var = float(np.dot(pi, np.diag(a)))
        port_var = float(pi @ a @ pi)
        return 0.5 * (weighted_var - port_var)

    def relative_covariance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Compute τ^π_{ij} = a_{ij} - a^π_i - a^π_j + a_{ππ}."""
        a_pi = a @ pi
        a_pipi = float(pi @ a_pi)
        tau: NDArray[np.float64] = a - np.add.outer(a_pi, a_pi) + a_pipi
        return tau

    def portfolio_variance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> float:
        """Compute π'aπ."""
        return float(pi @ a @ pi)

    def simulate_gbm_step(
        self,
        x: NDArray[np.float64],
        mu: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Exact GBM step: S(t+dt) = S(t) * exp((μ - σ²/2)dt + L·dW)."""
        a_diag = np.sum(cholesky**2, axis=1)
        log_increment = (mu - 0.5 * a_diag) * dt + cholesky @ dw
        result: NDArray[np.float64] = x * np.exp(log_increment)
        return result

    def diversity_weights(
        self,
        mu: NDArray[np.float64],
        p: float,
    ) -> NDArray[np.float64]:
        """Diversity-weighted portfolio: π_i = μ_i^p / Σ μ_j^p."""
        mu_p = mu**p
        result: NDArray[np.float64] = mu_p / np.sum(mu_p)
        return result

    def covariance_shrinkage(
        self,
        returns: NDArray[np.float64],
        shrinkage: float,
    ) -> NDArray[np.float64]:
        """Ledoit-Wolf style shrinkage toward scaled identity.

        Σ_shrunk = (1-α)·S + α·(tr(S)/n)·I
        """
        _n_obs, n_assets = returns.shape
        sample_cov = np.cov(returns, rowvar=False, ddof=1)
        target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        result: NDArray[np.float64] = (
            1.0 - shrinkage
        ) * sample_cov + shrinkage * target
        return result
