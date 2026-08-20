"""Atlas model and first-order rank-based models (BFK 2005).

The Atlas model is the canonical rank-based model in Stochastic Portfolio
Theory.  It generates a market where stock dynamics depend on *rank*
rather than identity, producing a stationary capital distribution with
Pareto tails — a striking match to empirical equity markets.

Classes
-------
AtlasModel
    Basic Atlas model where only the smallest stock receives positive growth.
FirstOrderModel
    General first-order rank-dependent growth and volatility model (BFK Eq. 1.6).

Mathematical References
-----------------------
- Atlas dynamics: BFK Eq. 1.1, 1.6–1.7
- Stability condition: BFK Eq. 1.5
- Ergodic property: BFK Prop. 2.3
- Ranked dynamics & local times: BFK Eq. 3.1–3.7
- Pareto exponents: BFK Eq. 4.3–4.4
- Certainty-equivalent weights: BFK Eq. 4.12–4.15
- Growth rate formulas: BFK Eq. 5.9–5.20
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._typing import StochasticProcess
from .base import MarketModel

__all__ = ["AtlasModel", "FirstOrderModel"]


# ---------------------------------------------------------------------------
# Atlas stochastic process (implements StochasticProcess protocol)
# ---------------------------------------------------------------------------


class _AtlasProcess:
    r"""StochasticProcess for the Atlas / first-order model.

    Log-capitalisation dynamics (BFK Eq. 1.6):

    .. math::
        dY_i(t) = \bigl[\gamma + g_{r_i(t)}\bigr]\,dt
                  + \sigma_{r_i(t)}\,dW_i(t)

    where :math:`r_i(t)` is the rank of stock *i* at time *t*
    (1 = largest, *n* = smallest) and :math:`Y_i = \log X_i`.

    The process operates in **log-capitalisation** space and uses
    Euler–Maruyama steps with rank assignment at each step.
    """

    def __init__(
        self,
        n: int,
        gamma: float,
        g: NDArray[np.float64],
        sigma: NDArray[np.float64],
        x0: NDArray[np.float64],
    ) -> None:
        self._n = n
        self._gamma = gamma
        self._g = g
        self._sigma = sigma
        self._x0 = x0.copy()

    def size(self) -> int:
        return self._n

    def factors(self) -> int:
        return self._n

    def initial_values(self) -> NDArray[np.float64]:
        return self._x0.copy()

    def _ranks(self, x: NDArray[np.float64]) -> NDArray[np.intp]:
        """Rank stocks 1..n (1 = largest) from log-caps."""
        order = np.argsort(-x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self._n)
        return ranks

    def drift(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ranks = self._ranks(x)
        return self._gamma + self._g[ranks]

    def diffusion(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ranks = self._ranks(x)
        return np.diag(self._sigma[ranks])

    def evolve(
        self,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Euler–Maruyama step in log-cap space."""
        mu = self.drift(t0, x0)
        sigma_diag = np.diag(self.diffusion(t0, x0))
        return x0 + mu * dt + sigma_diag * dw


# ---------------------------------------------------------------------------
# FirstOrderModel — general rank-dependent parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstOrderModel(MarketModel):
    r"""General first-order rank-based model (BFK Eq. 1.6).

    Log-capitalisation dynamics:

    .. math::
        dY_i(t) = [\gamma + g_{r_i(t)}]\,dt + \sigma_{r_i(t)}\,dW_i(t)

    where :math:`Y_i = \log X_i`, and the rank-dependent parameters
    :math:`g_k`, :math:`\sigma_k` for :math:`k = 1, \dots, n` satisfy
    the **stability condition** (BFK Eq. 1.5):

    .. math::
        g_1 + \cdots + g_k < 0 \quad \text{for } k = 1, \dots, n-1,
        \qquad
        \sum_{k=1}^{n} g_k = 0

    Parameters
    ----------
    n : int
        Number of stocks.
    gamma : float
        Common drift parameter (long-term market growth rate).
    g : ndarray of shape (n,)
        Rank-dependent growth-rate increments satisfying the stability
        condition.
    sigma : ndarray of shape (n,)
        Rank-dependent volatilities (positive).
    """

    n: int
    gamma: float
    g: NDArray[np.float64]
    sigma: NDArray[np.float64]
    _local_time_rates: NDArray[np.float64] = field(init=False, repr=False)
    _pareto_exp: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require(self.n >= 2, f"Need at least 2 stocks, got {self.n}")
        require(len(self.g) == self.n, "g length must equal n")
        require(len(self.sigma) == self.n, "sigma length must equal n")
        require(
            bool(np.all(self.sigma > 0)),
            "All volatilities must be positive",
        )

        cumsum = np.cumsum(self.g)
        require(
            bool(np.all(cumsum[:-1] < 0)),
            f"Stability violated: partial sums g_1+…+g_k must be < 0 "
            f"for k < n. Got cumsum = {cumsum[:-1]}",
        )
        require(
            bool(np.isclose(cumsum[-1], 0.0, atol=1e-10)),
            f"Stability violated: g must sum to 0. Got sum = {cumsum[-1]:.2e}",
        )

        # Local time rates λ_{k,k+1} = −2(g_1+…+g_k)  (BFK Eq. 3.7)
        object.__setattr__(self, "_local_time_rates", -2.0 * cumsum[:-1])

        # Pareto exponents r_k = −4(g_1+…+g_k)/(σ²_k + σ²_{k+1}) (BFK Eq. 4.3)
        sigma_sq = self.sigma**2
        denom = sigma_sq[:-1] + sigma_sq[1:]
        object.__setattr__(self, "_pareto_exp", -4.0 * cumsum[:-1] / denom)

    @property
    def n_assets(self) -> int:
        return self.n

    @property
    def log_space_process(self) -> bool:
        """Atlas models operate in log-capitalisation space."""
        return True

    # ----- Analytical results (BFK §3–5) -----

    def local_time_rates(self) -> NDArray[np.float64]:
        r"""Asymptotic local-time accumulation rates (BFK Eq. 3.7).

        .. math::
            \lambda_{k,k+1} = -2\,(g_1 + \cdots + g_k) > 0,
            \quad k = 1, \dots, n-1

        Returns
        -------
        ndarray of shape (n−1,)
        """
        return self._local_time_rates.copy()

    def pareto_exponents(self) -> NDArray[np.float64]:
        r"""Pareto exponents for successive ranked-weight ratios (BFK Eq. 4.3).

        .. math::
            r_k = \frac{-4(g_1 + \cdots + g_k)}
                       {\sigma^2_k + \sigma^2_{k+1}}

        In steady state :math:`P[\mu_{(k)}/\mu_{(k+1)} > y] \to y^{-r_k}`.

        Returns
        -------
        ndarray of shape (n−1,)
        """
        return self._pareto_exp.copy()

    def certainty_equivalent_weights(self) -> NDArray[np.float64]:
        r"""Certainty-equivalent approximation of ranked market weights (BFK Eq. 4.12–4.15).

        .. math::
            M^{\text{CE}}_k \approx
            \frac{\exp(\rho_{n-1} + \cdots + \rho_k)}
                 {\sum_j \exp(\rho_{n-1} + \cdots + \rho_j)}

        where :math:`\rho_k = (\sigma^2_k + \sigma^2_{k+1}) / (2 \lambda_{k,k+1})`.

        Returns
        -------
        ndarray of shape (n,)
            Approximate steady-state ranked market weights (summing to 1).
        """
        sigma_sq = self.sigma**2
        lam = self._local_time_rates  # (n-1,)
        # BFK Eq. 4.12: ρ_k = (σ²_k + σ²_{k+1}) / (2·λ_{k,k+1})
        rho = (sigma_sq[:-1] + sigma_sq[1:]) / (2.0 * lam)  # (n-1,)

        log_M = np.zeros(self.n)
        for k in range(self.n - 1):
            log_M[k] = np.sum(rho[k:])
        # log_M[n-1] = 0 (empty sum, smallest stock)

        log_M -= np.max(log_M)  # numerical stability
        M_CE = np.exp(log_M)
        return M_CE / np.sum(M_CE)

    def equal_weighted_excess_growth_rate(self) -> float:
        r"""γ*_η = (n−1)/(2n²) · Σ_k σ²_k   (BFK Eq. 5.14).

        Returns
        -------
        float
        """
        return (self.n - 1) / (2.0 * self.n**2) * float(np.sum(self.sigma**2))

    def equal_weighted_growth_rate(self) -> float:
        r"""G^η(n) = γ + γ*_η   (BFK Eq. 5.14–5.15).

        Returns
        -------
        float
        """
        return self.gamma + self.equal_weighted_excess_growth_rate()

    def market_growth_rate(self) -> float:
        r"""G^μ(n) = γ   (BFK Eq. 5.10).

        Returns
        -------
        float
        """
        return self.gamma

    def diversity_weighted_excess_growth(
        self,
        p: float,
        M: NDArray[np.float64] | None = None,
    ) -> float:
        r"""Long-term excess growth rate of the diversity-weighted portfolio.

        Computes the ergodic (time-averaged) excess growth rate
        G^{ϑ(p)}_*(n) from BFK Eq. 5.19–5.20:

        .. math::
            G^{\vartheta(p)}_*(n) = \frac{g}{p}\left[
                1 - n \cdot \frac{M_n^p}{\sum_{k=1}^n M_k^p}
            \right]

        This is the long-run average over the stationary distribution of
        ranked market weights, NOT the instantaneous excess growth rate
        at a single time point.  The instantaneous formula
        γ*_π = ½ Σ π_k(1−π_k)σ²_k applies at a fixed t; this method
        gives the time-averaged equivalent under ergodicity.

        Parameters
        ----------
        p : float
            Diversity parameter 0 < p < 1.
        M : ndarray of shape (n,), optional
            Steady-state ranked market weights.  If ``None``, the certainty-
            equivalent approximation (BFK Eq. 4.12–4.15) is used.

        Returns
        -------
        float
            Long-term excess growth rate G^{ϑ(p)}_*(n).

        References
        ----------
        BFK Eq. 5.19–5.20
        """
        require(0 < p < 1, f"p must be in (0, 1), got {p}")
        if M is None:
            M = self.certainty_equivalent_weights()
        sigma_sq = self.sigma**2
        M_p = M**p
        D_p = float(np.sum(M_p))
        pi = M_p / D_p
        gamma_star = 0.5 * float(np.sum(pi * (1.0 - pi) * sigma_sq))
        return gamma_star

    # ----- MarketModel interface -----

    def drift_rates(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Rank-dependent growth rates in log-cap space.

        Returns γ_i(t) = γ + g_{rank(i)}.

        Since the BFK model is defined in log-capitalisation space where
        dY_i = [γ + g_{r_i}] dt + σ_{r_i} dW_i, the drift γ + g_{r_i}
        is already the growth rate (not the rate of return). No Ito
        correction is needed.
        """
        order = np.argsort(-x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.n)
        return self.gamma + self.g[ranks]

    def covariance_rate(
        self,
        t: float,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Diagonal covariance (independent Brownian motions) with rank-dependent σ.

        In the Atlas model stocks are driven by *independent* Brownian
        motions, so the covariance matrix is diagonal with entries
        σ²_{rank(i)}.
        """
        order = np.argsort(-x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.n)
        sigma_sq = self.sigma[ranks] ** 2
        return np.diag(sigma_sq)

    def to_stochastic_process(
        self,
        x0: NDArray[np.float64],
    ) -> StochasticProcess:
        """Return an ``_AtlasProcess`` in log-cap space."""
        require(len(x0) == self.n, "x0 length mismatch")
        require(bool(np.all(x0 > 0)), "Initial values must be positive")
        log_x0 = np.log(x0)
        proc = _AtlasProcess(
            n=self.n,
            gamma=self.gamma,
            g=self.g.copy(),
            sigma=self.sigma.copy(),
            x0=log_x0,
        )
        return proc


# ---------------------------------------------------------------------------
# AtlasModel — the basic Atlas specialisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasModel(FirstOrderModel):
    r"""Basic Atlas model (BFK Eq. 1.7).

    A special case of :class:`FirstOrderModel` where only the *smallest*
    stock (rank *n*) receives a positive growth-rate push:

    .. math::
        g_k = -g \quad k = 1, \dots, n-1, \qquad g_n = (n-1)\,g

    and all ranks share a common volatility σ.

    Parameters
    ----------
    n : int
        Number of stocks.
    gamma : float
        Common drift parameter (= market portfolio long-term growth rate).
    g_param : float
        Atlas growth parameter *g* > 0.  The smallest stock's growth push
        is (n−1)·g, ensuring the stability condition Σg_k = 0.
    sigma_param : float
        Common volatility σ > 0 for all ranks.
    """

    g_param: float = field(default=0.0)
    sigma_param: float = field(default=0.0)

    def __init__(
        self,
        n: int,
        gamma: float,
        g_param: float,
        sigma_param: float,
    ) -> None:
        require(g_param > 0, f"g_param must be positive, got {g_param}")
        require(sigma_param > 0, f"sigma_param must be positive, got {sigma_param}")

        g_vec = np.full(n, -g_param)
        g_vec[-1] = (n - 1) * g_param
        sigma_vec = np.full(n, sigma_param)

        object.__setattr__(self, "g_param", g_param)
        object.__setattr__(self, "sigma_param", sigma_param)
        # Initialise parent
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "g", g_vec)
        object.__setattr__(self, "sigma", sigma_vec)
        # Trigger parent validation and derived attributes
        FirstOrderModel.__post_init__(self)

    # ----- Atlas-specific analytical shortcuts -----

    def pareto_exponent(self) -> float:
        r"""Common Pareto exponent for constant-σ Atlas (BFK §4).

        When all σ_k = σ, every :math:`r_k` simplifies to:

        .. math::
            r_k = \frac{2\,g_{\text{param}}\,k}{\sigma^2}

        For the basic Atlas the *first* exponent (k=1) is:

        .. math::
            r_1 = \frac{2\,g}{\sigma^2}

        Returns
        -------
        float
            The first Pareto exponent r_1.
        """
        return 2.0 * self.g_param / self.sigma_param**2

    def zipf_exponent(self) -> float:
        r"""Zipf exponent α = σ²/(2g) governing ranked market weights (BFK Eq. 4.17).

        In the constant-σ Atlas the certainty-equivalent ranked weights
        follow an approximate Zipf/power law:

        .. math::
            M^{\text{CE}}_k \propto k^{-\alpha},
            \quad \alpha = \frac{\sigma^2}{2g}

        Returns
        -------
        float
        """
        return self.sigma_param**2 / (2.0 * self.g_param)
