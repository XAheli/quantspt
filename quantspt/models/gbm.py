"""Correlated Geometric Brownian Motion market model.

The baseline market model in SPT: *n* stocks follow a multivariate GBM
with constant drift and covariance parameters.  This wraps the core
:class:`~quantspt.core.processes.CorrelatedGBM` process and adds the
market-level interface required by :class:`MarketModel`.

Mathematical References
-----------------------
- GBM dynamics: dS_i = μ_i S_i dt + S_i Σ_ν L_{iν} dW_ν
- Growth rate γ_i = μ_i − a_{ii}/2 (Itô's lemma)
- Excess growth rate γ*_π: F&K Survey Eq. 1.13, FKK Eq. 2.8
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._typing import StochasticProcess
from ..core.processes import CorrelatedGBM
from .base import MarketModel

__all__ = ["CorrelatedGBMMarket"]


@dataclass(frozen=True)
class CorrelatedGBMMarket(MarketModel):
    r"""Correlated GBM market model.

    Each stock follows geometric Brownian motion:

    .. math::
        dS_i = \mu_i S_i\,dt + S_i \sum_\nu L_{i\nu}\,dW_\nu

    where :math:`L` is the Cholesky factor of the covariance matrix
    :math:`a = LL^T`.

    This is the simplest non-trivial model in SPT and serves as the
    analytical benchmark: many quantities have closed-form expressions.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Drift rates (rates of return) for each stock.
    cov : ndarray of shape (n, n)
        Instantaneous covariance matrix.  Must be symmetric PSD.

    Notes
    -----
    The individual stock growth rates are:

    .. math::
        \gamma_i = \mu_i - \tfrac{1}{2}a_{ii}

    which are *constant* in this model (time- and state-independent).
    """

    mu: NDArray[np.float64]
    cov: NDArray[np.float64]
    _growth_rates: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.mu)
        require(self.cov.shape == (n, n), "Covariance shape mismatch")
        require(
            bool(np.allclose(self.cov, self.cov.T)),
            "Covariance must be symmetric",
        )
        eigenvalues = np.linalg.eigvalsh(self.cov)
        require(
            bool(np.all(eigenvalues >= -1e-10)),
            f"Covariance must be PSD, min eigenvalue = {eigenvalues[0]:.2e}",
        )
        gamma = self.mu - 0.5 * np.diag(self.cov)
        object.__setattr__(self, "_growth_rates", gamma)

    @property
    def n_assets(self) -> int:
        return len(self.mu)

    def drift_rates(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Constant growth rates γ_i = μ_i − a_{ii}/2."""
        return self._growth_rates.copy()

    def covariance_rate(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Constant covariance matrix."""
        return self.cov.copy()

    def to_stochastic_process(
        self,
        x0: NDArray[np.float64],
    ) -> StochasticProcess:
        """Wrap as a :class:`CorrelatedGBM` for simulation."""
        require(len(x0) == self.n_assets, "x0 length mismatch")
        require(bool(np.all(x0 > 0)), "Initial values must be positive")
        proc = CorrelatedGBM(
            mu=self.mu.copy(),
            cov=self.cov.copy(),
            x0=x0.copy(),
        )
        return proc
