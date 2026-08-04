"""Market model protocol — the common interface for all SPT market models.

Every concrete market model (GBM, Atlas, volatility-stabilised, etc.) implements
this protocol so that simulators and analytics can consume any model uniformly.

A ``MarketModel`` supplies:

- **n_assets**: the number of stocks in the market
- **drift_rates(t, x)**: individual stock growth rates γ_i
- **covariance_rate(t, x)**: instantaneous covariance matrix a_{ij}
- **to_stochastic_process()**: convert to a ``StochasticProcess`` for simulation

Mathematical References
-----------------------
- Growth rates γ_i: F&K Survey Eq. 1.11
- Covariance rate a_{ij}: F&K Survey Eq. 1.3, FKK Eq. 2.5
- Non-degeneracy condition: FKK Eq. 2.3
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .._typing import StochasticProcess

__all__ = ["MarketModel"]


class MarketModel(ABC):
    r"""Abstract base class for continuous-time market models.

    A market model specifies the joint dynamics of *n* stock capitalizations
    via drift and covariance coefficients.  Every model can be converted to
    a :class:`~quantspt._typing.StochasticProcess` for Monte-Carlo simulation.

    Subclasses must implement:

    - :attr:`n_assets`
    - :meth:`drift_rates`
    - :meth:`covariance_rate`
    - :meth:`to_stochastic_process`
    """

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of stocks in the market."""
        ...

    @abstractmethod
    def drift_rates(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Instantaneous growth rates γ_i(t) for each stock.

        .. math::
            \gamma_i(t) = b_i(t) - \tfrac{1}{2}a_{ii}(t)

        where b_i is the rate of return and a_{ii} the variance rate.

        Parameters
        ----------
        t : float
            Current time.
        x : ndarray of shape (n,)
            Current stock capitalizations (or log-caps, depending on model).

        Returns
        -------
        ndarray of shape (n,)
            Growth rates [γ_1, …, γ_n].

        References
        ----------
        F&K Survey Eq. 1.11
        """
        ...

    @abstractmethod
    def covariance_rate(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Instantaneous covariance rate matrix a_{ij}(t).

        .. math::
            a_{ij}(t) = \sum_{\nu} \sigma_{i\nu}(t)\,\sigma_{j\nu}(t)

        Parameters
        ----------
        t : float
            Current time.
        x : ndarray of shape (n,)
            Current stock capitalizations.

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric positive semi-definite covariance matrix.

        References
        ----------
        F&K Survey Eq. 1.3, FKK Eq. 2.5
        """
        ...

    @abstractmethod
    def to_stochastic_process(
        self,
        x0: NDArray[np.float64],
    ) -> StochasticProcess:
        """Convert to a :class:`StochasticProcess` suitable for simulation.

        Parameters
        ----------
        x0 : ndarray of shape (n,)
            Initial stock capitalizations.  Must be positive.

        Returns
        -------
        StochasticProcess
            Object implementing ``size()``, ``factors()``, ``drift()``,
            ``diffusion()``, ``evolve()``, and ``initial_values()``.
        """
        ...

    def market_weights(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute market-capitalisation weights from price levels.

        .. math::
            \mu_i(t) = \frac{X_i(t)}{\sum_j X_j(t)}

        Parameters
        ----------
        x : ndarray of shape (n,)
            Current stock capitalizations.

        Returns
        -------
        ndarray of shape (n,)
            Market weights summing to 1.
        """
        total = np.sum(x)
        return x / total
