"""Volatility-stabilised market model.

In the volatility-stabilised model the diffusion coefficient of each stock
is inversely proportional to the square root of its market weight, making
smaller stocks more volatile while keeping the aggregate market volatility
bounded.  This model is intrinsically *diverse* and contains explicit
relative-arbitrage opportunities.

Mathematical References
-----------------------
- Volatility-stabilised dynamics: Lukacs §12, F&K Survey §14
- Market excess growth rate: γ*_μ = σ²(n−1)/(2n)
- Connection to squared Bessel processes (BESQ): Lukacs §12
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._typing import StochasticProcess
from .base import MarketModel

__all__ = ["VolatilityStabilizedMarket"]


class _VolStabProcess:
    r"""StochasticProcess for the volatility-stabilised market.

    Capitalisation dynamics:

    .. math::
        dX_i(t) = X_i(t)\,\frac{\sigma^2}{2}\,dt
                  + X_i(t)\,\frac{\sigma}{\sqrt{n\,\mu_i(t)}}\,dW_i(t)

    where :math:`\mu_i(t) = X_i(t) / \sum_j X_j(t)` is the market weight.

    The process uses Euler–Maruyama in **log-capitalisation** space.
    """

    def __init__(
        self,
        n: int,
        sigma: float,
        x0: NDArray[np.float64],
    ) -> None:
        self._n = n
        self._sigma = sigma
        self._x0 = x0.copy()

    def size(self) -> int:
        return self._n

    def factors(self) -> int:
        return self._n

    def initial_values(self) -> NDArray[np.float64]:
        return self._x0.copy()

    def _weights_from_log_caps(self, log_x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Market weights from log-capitalisations (numerically stable)."""
        log_x_shifted = log_x - np.max(log_x)
        caps = np.exp(log_x_shifted)
        return caps / np.sum(caps)

    def drift(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        mu = self._weights_from_log_caps(x)
        sigma_sq_over_n = self._sigma**2 / self._n
        return np.full(self._n, 0.5 * self._sigma**2) - 0.5 * sigma_sq_over_n / mu

    def diffusion(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        mu = self._weights_from_log_caps(x)
        sigma_i = self._sigma / np.sqrt(self._n * mu)
        return np.diag(sigma_i)

    def evolve(
        self,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Euler–Maruyama step in log-cap space."""
        mu_vec = self.drift(t0, x0)
        sigma_diag = np.diag(self.diffusion(t0, x0))
        return x0 + mu_vec * dt + sigma_diag * dw


@dataclass(frozen=True)
class VolatilityStabilizedMarket(MarketModel):
    r"""Volatility-stabilised market model (Lukacs §12, F&K Survey §14).

    Each stock's volatility is inversely proportional to the square root
    of its market weight:

    .. math::
        \sigma_i(t) = \frac{\sigma}{\sqrt{n\,\mu_i(t)}}

    so that smaller stocks are more volatile.  The market weight dynamics
    are related to squared Bessel processes, and the model is:

    - **Diverse**: the market is intrinsically diverse.
    - **Contains relative arbitrage**: the diversity-weighted portfolio
      beats the market over a sufficiently long horizon.
    - **Analytically tractable**: the market excess growth rate is
      γ*_μ = σ²(n−1)/(2n), independent of the weight configuration.

    Parameters
    ----------
    n : int
        Number of stocks.
    sigma : float
        Base volatility parameter σ > 0.

    References
    ----------
    Lukacs §12, F&K Survey §14
    """

    n: int
    sigma: float

    def __post_init__(self) -> None:
        require(self.n >= 2, f"Need at least 2 stocks, got {self.n}")
        require(self.sigma > 0, f"sigma must be positive, got {self.sigma}")

    @property
    def n_assets(self) -> int:
        return self.n

    @property
    def log_space_process(self) -> bool:
        """Volatility-stabilised models operate in log-capitalisation space."""
        return True

    def stock_variance(self, mu_i: float) -> float:
        r"""Variance rate for a stock with market weight μ_i.

        .. math::
            a_{ii} = \frac{\sigma^2}{n\,\mu_i}

        Parameters
        ----------
        mu_i : float
            Market weight of the stock.

        Returns
        -------
        float

        References
        ----------
        Lukacs §12
        """
        require(mu_i > 0, f"Weight must be positive, got {mu_i}")
        return self.sigma**2 / (self.n * mu_i)

    def market_excess_growth_rate(self) -> float:
        r"""Excess growth rate of the market portfolio.

        .. math::
            \gamma^*_\mu = \frac{\sigma^2 (n - 1)}{2n}

        This is *constant* — independent of the weight distribution.

        Returns
        -------
        float

        References
        ----------
        Lukacs §12, F&K Survey §14
        """
        return self.sigma**2 * (self.n - 1) / (2.0 * self.n)

    def drift_rates(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Growth rates γ_i = (rate of return) − a_{ii}/2.

        In the vol-stabilised model all stocks have the same rate of return
        σ²/2, so γ_i = σ²/2 − a_{ii}/2 = σ²/2 − σ²/(2nμ_i).
        """
        mu = x / np.sum(x)
        a_ii = self.sigma**2 / (self.n * mu)
        return np.full(self.n, 0.5 * self.sigma**2) - 0.5 * a_ii

    def covariance_rate(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Diagonal covariance with entries σ²/(n·μ_i)."""
        mu = x / np.sum(x)
        a_ii = self.sigma**2 / (self.n * mu)
        return np.diag(a_ii)

    def to_stochastic_process(
        self,
        x0: NDArray[np.float64],
    ) -> StochasticProcess:
        """Create log-cap process for simulation."""
        require(len(x0) == self.n, "x0 length mismatch")
        require(bool(np.all(x0 > 0)), "Initial values must be positive")
        log_x0 = np.log(x0)
        proc = _VolStabProcess(n=self.n, sigma=self.sigma, x0=log_x0)
        return proc
