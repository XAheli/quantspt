"""Functionally Generated Portfolios (FGP) — the crown jewel of SPT.

A C² positive function G on the open simplex mechanically produces a
portfolio whose performance is completely characterized by the master
formula. This module implements the FGP framework and provides the
standard generating functions from the literature.

Mathematical References
-----------------------
- FGP weight formula: F&K Survey Eq. 11.1, Lukacs Lectures §11
- Master formula: F&K Survey Eq. 11.2
- Drift process: F&K Survey Eq. 11.3
- Diversity generator G_p: F&K Survey Remark 11.1 (Example 3)
- Entropy generator: F&K Survey Eq. 11.5, Lukacs Lectures §11
- Modified entropy H_c: F&K Survey Eq. 11.6-11.7
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "GeneratingFunction",
    "DiversityGenerator",
    "EntropyGenerator",
    "ModifiedEntropyGenerator",
    "InverseVolatilityGenerator",
    "CustomGenerator",
    "fernholz_weights",
    "drift_process",
]


class GeneratingFunction(ABC):
    r"""Portfolio Generating Function G: Δ_n → ℝ₊.

    Given a C² positive function G on the open simplex, the Fernholz formula
    produces portfolio weights:

    .. math::
        \pi_i = \left[D_i \log G(\mu) + 1
                 - \sum_k \mu_k D_k \log G(\mu)\right] \mu_i

    And the master formula gives the complete performance decomposition:

    .. math::
        \log\frac{V^{\pi}(T)}{V^{\mu}(T)}
        = \log\frac{G(\mu(T))}{G(\mu(0))} + \int_0^T g(t)\,dt

    where the drift process g(t) is:

    .. math::
        g(t) = -\frac{1}{2G(\mu)} \sum_{i,j}
               D^2_{ij} G(\mu)\,\mu_i\,\mu_j\,\tau^{\mu}_{ij}

    References
    ----------
    F&K Survey Eq. 11.1-11.3, Lukacs Lectures §11
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display and logging."""
        ...

    @abstractmethod
    def __call__(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G(μ)."""
        ...

    @abstractmethod
    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute ∇ log G(μ), i.e., [D_k log G(μ)]_{k=1}^n.

        Must be implemented analytically for numerical stability.
        """
        ...

    @abstractmethod
    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute Hessian D²G(μ) ∈ ℝ^{n×n}.

        Used in drift process computation.
        """
        ...

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Compute portfolio weights via Fernholz formula.

        .. math::
            \pi_i = \left[D_i \log G(\mu)
                     + 1 - \sum_k \mu_k D_k \log G(\mu)\right] \mu_i

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Market weights (positive, sum to 1).

        Returns
        -------
        ndarray of shape (n,)
            Portfolio weights.

        References
        ----------
        F&K Survey Eq. 11.1
        """
        return fernholz_weights(self.log_gradient(mu), mu)

    def drift(
        self,
        mu: NDArray[np.float64],
        tau_mu: NDArray[np.float64],
    ) -> float:
        r"""Compute drift g(t) from master formula.

        .. math::
            g(t) = -\frac{1}{2G(\mu)} \sum_{i,j}
                   D^2_{ij}G(\mu)\,\mu_i\,\mu_j\,\tau^{\mu}_{ij}

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Market weights.
        tau_mu : ndarray of shape (n, n)
            Relative covariance matrix of the market portfolio.

        Returns
        -------
        float
            Drift process value. Positive drift means the FGP outperforms.

        References
        ----------
        F&K Survey Eq. 11.3
        """
        return drift_process(self, mu, tau_mu)


class DiversityGenerator(GeneratingFunction):
    r"""Diversity-weighted generating function G_p(μ) = (Σ μ_i^p)^{1/p}.

    For p ∈ (0, 1), this produces a portfolio that overweights small stocks.
    It is 1-homogeneous, so weights automatically sum to 1.

    The resulting portfolio weights are:

    .. math::
        \pi_i^{(p)} = \frac{\mu_i^p}{\sum_j \mu_j^p}

    And the drift process is non-negative when the market is weakly diverse:

    .. math::
        g_p(t) = \frac{p(1-p)}{2} \cdot
                 \frac{\sum_{i,j} \mu_i^p \mu_j^p
                       (\tau^{\mu}_{ii} + \tau^{\mu}_{jj} - 2\tau^{\mu}_{ij})}
                      {2(\sum_k \mu_k^p)^2}

    Parameters
    ----------
    p : float
        Diversity parameter, p ∈ (0, 1). Smaller p gives more tilt to
        small stocks.

    References
    ----------
    F&K Survey Remark 11.1 (Example 3), FKK Eq. 4.4
    """

    def __init__(self, p: float) -> None:
        require(0 < p < 1, f"Diversity parameter must be in (0, 1), got {p}")
        self._p = p

    @property
    def p(self) -> float:
        """Diversity exponent."""
        return self._p

    @property
    def name(self) -> str:
        return f"Diversity(p={self._p:.3f})"

    def __call__(self, mu: NDArray[np.float64]) -> float:
        """G_p(μ) = (Σ μ_i^p)^{1/p}."""
        return float(np.sum(mu**self._p)) ** (1.0 / self._p)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D_k log G_p = (p-1) · log μ_k ... actually computed directly.

        D_k log G_p(μ) = μ_k^{p-1} / Σ_j μ_j^p
                       = (p-1) log μ_k ... NO, that's wrong.

        Correct derivation:
            log G_p = (1/p) log(Σ μ_i^p)
            D_k log G_p = (1/p) · p μ_k^{p-1} / Σ μ_j^p
                        = μ_k^{p-1} / Σ μ_j^p
        """
        mu_p = mu ** (self._p - 1.0)
        S = np.sum(mu**self._p)
        return mu_p / S

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D²_{ij} G_p(μ).

        G_p = S^{1/p} where S = Σ μ_k^p.

        D_i G_p = S^{1/p - 1} μ_i^{p-1}

        D²_{ij} G_p = S^{1/p - 2} [(1/p - 1) μ_i^{p-1} μ_j^{p-1}
                       + S (p-1) μ_i^{p-2} δ_{ij}]
        """
        p = self._p
        S = float(np.sum(mu**p))
        mu_pm1 = mu ** (p - 1.0)

        term1_coeff = S ** (1.0 / p - 2.0) * (1.0 / p - 1.0)
        term1 = term1_coeff * np.outer(mu_pm1, mu_pm1)

        term2_coeff = S ** (1.0 / p - 1.0) * (p - 1.0)
        term2 = term2_coeff * np.diag(mu ** (p - 2.0))

        return term1 + term2

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Diversity-weighted portfolio: π_i = μ_i^p / Σ μ_j^p.

        Since G_p is 1-homogeneous, the Fernholz formula simplifies to
        this direct ratio form.

        References
        ----------
        FKK Eq. 4.4
        """
        mu_p = mu**self._p
        return mu_p / np.sum(mu_p)


class EntropyGenerator(GeneratingFunction):
    r"""Entropy-based generating function G(μ) = exp(H(μ)).

    .. math::
        G(\mu) = \exp\left(-\sum_i \mu_i \log \mu_i\right)
               = \prod_i \mu_i^{-\mu_i}

    This is NOT 1-homogeneous, so the Fernholz formula must be used
    for weight computation (not a simple ratio).

    The resulting weights are:

    .. math::
        \pi_i = \mu_i \left[-\log \mu_i - H(\mu)\right] + \mu_i
              = -\mu_i \log \mu_i

    Wait — let's derive properly:
        D_k log G = D_k H = -(1 + log μ_k)
        Σ μ_k D_k log G = Σ μ_k (-(1 + log μ_k)) = -(1 + Σ μ_k log μ_k) = -(1 - H)
        π_i = [-(1 + log μ_i) + 1 - (-(1 - H))] μ_i
            = [-(1 + log μ_i) + 1 + 1 - H] μ_i
            = [1 - H - log μ_i] μ_i

    References
    ----------
    F&K Survey Eq. 11.5, Lukacs Lectures §11
    """

    @property
    def name(self) -> str:
        return "Entropy"

    def __call__(self, mu: NDArray[np.float64]) -> float:
        """G(μ) = exp(H(μ)) = exp(-Σ μ_i log μ_i)."""
        H = -float(np.sum(mu * np.log(mu)))
        return np.exp(H)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D_k log G = D_k H = -(1 + log μ_k)."""
        return -(1.0 + np.log(mu))

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D²_{ij} G(μ) where G = exp(H).

        D_i G = G · D_i H = G · (-(1 + log μ_i))
        D²_{ij} G = G · [(-(1 + log μ_i))(-(1 + log μ_j)) - δ_{ij}/μ_i]
                  = G · [(1 + log μ_i)(1 + log μ_j) - δ_{ij}/μ_i]
        """
        G_val = self(mu)
        log_terms = 1.0 + np.log(mu)
        H = np.outer(log_terms, log_terms) - np.diag(1.0 / mu)
        return np.asarray(G_val * H, dtype=np.float64)


class ModifiedEntropyGenerator(GeneratingFunction):
    r"""Modified entropy generating function H_c(μ) = c + H(μ).

    .. math::
        H_c(\mu) = c - \sum_i \mu_i \log \mu_i

    where c > 0 is chosen so that H_c > 0 on the simplex.

    This is the generating function actually used in the Fernholz papers
    for proving arbitrage results via the entropy. Unlike exp(H), this is
    the direct function whose drift analysis yields the sufficient
    intrinsic volatility condition.

    Parameters
    ----------
    c : float
        Shift parameter, must satisfy c > 0 (typically c ≥ some safety margin
        to ensure H_c > 0 everywhere on the simplex).

    References
    ----------
    F&K Survey Eq. 11.6-11.7
    """

    def __init__(self, c: float) -> None:
        require(c > 0, f"Shift parameter c must be positive, got {c}")
        self._c = c

    @property
    def c(self) -> float:
        """Shift parameter."""
        return self._c

    @property
    def name(self) -> str:
        return f"ModifiedEntropy(c={self._c:.3f})"

    def __call__(self, mu: NDArray[np.float64]) -> float:
        """H_c(μ) = c + H(μ) = c - Σ μ_i log μ_i."""
        H = -float(np.sum(mu * np.log(mu)))
        return self._c + H

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D_k log H_c = D_k H_c / H_c = -(1 + log μ_k) / H_c."""
        Hc = self(mu)
        return -(1.0 + np.log(mu)) / Hc

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""D²_{ij} H_c(μ).

        H_c is linear in μ via the entropy terms:
            D_i H_c = -(1 + log μ_i)
            D²_{ij} H_c = -δ_{ij} / μ_i
        """
        return -np.diag(1.0 / mu)

    def drift(
        self,
        mu: NDArray[np.float64],
        tau_mu: NDArray[np.float64],
    ) -> float:
        r"""Drift of H_c: g_c(t) = γ*_μ(t) / H_c(μ(t)).

        This elegant result follows because the Hessian is -diag(1/μ_i):

        .. math::
            g_c = -\frac{1}{2 H_c} \sum_{ij} D^2_{ij} H_c \cdot
                  \mu_i \mu_j \tau^{\mu}_{ij}
                = \frac{1}{2 H_c} \sum_i \mu_i \tau^{\mu}_{ii}
                = \frac{\gamma^*_{\mu}}{H_c}

        References
        ----------
        F&K Survey Eq. 11.7, Lukacs Lectures Eq. 11.5
        """
        from .growth_rates import excess_growth_rate_from_tau

        Hc = self(mu)
        gamma_star_mu = excess_growth_rate_from_tau(mu, tau_mu)
        return gamma_star_mu / Hc


@dataclass(frozen=True)
class InverseVolatilityGenerator(GeneratingFunction):
    r"""Inverse-volatility weighted generating function.

    Weights stocks inversely proportional to their variance rate:

    .. math::
        \pi_i = \frac{1/a_{ii}}{\sum_j 1/a_{jj}} \cdot \mu_i^0

    This is not strictly an FGP (it depends on a, not just μ), but is
    included for practical completeness. Uses numerical derivatives.

    Parameters
    ----------
    variances : ndarray of shape (n,)
        Diagonal elements of the covariance rate matrix.
    """

    variances: NDArray[np.float64]

    @property
    def name(self) -> str:
        return "InverseVolatility"

    def __call__(self, mu: NDArray[np.float64]) -> float:
        inv_var = 1.0 / self.variances
        return float(np.sum(inv_var * mu))

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        inv_var = 1.0 / self.variances
        G_val = float(np.sum(inv_var * mu))
        return inv_var / G_val

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(mu)
        return np.zeros((n, n))

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse-volatility weights (independent of market weights)."""
        inv_var = 1.0 / self.variances
        return inv_var / np.sum(inv_var)


class CustomGenerator(GeneratingFunction):
    """User-defined generating function with numerical derivatives.

    For rapid prototyping and research. Analytical derivatives are strongly
    preferred for production use (see DiversityGenerator, EntropyGenerator).

    Parameters
    ----------
    func : callable
        G(μ) → float. Must be C² and positive on the open simplex.
    name_str : str
        Display name.
    h : float
        Step size for finite differences.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        name_str: str = "Custom",
        h: float = 1e-7,
    ) -> None:
        self._func = func
        self._name = name_str
        self._h = h

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, mu: NDArray[np.float64]) -> float:
        return float(self._func(mu))

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Central-difference approximation of ∇ log G."""
        n = len(mu)
        grad = np.zeros(n)
        for k in range(n):
            mu_plus = mu.copy()
            mu_plus[k] += self._h
            mu_minus = mu.copy()
            mu_minus[k] -= self._h
            grad[k] = (np.log(self(mu_plus)) - np.log(self(mu_minus))) / (2 * self._h)
        return grad

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Second-order finite difference Hessian of G."""
        n = len(mu)
        H = np.zeros((n, n))
        G0 = self(mu)
        h = self._h
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mu_p = mu.copy()
                    mu_p[i] += h
                    mu_m = mu.copy()
                    mu_m[i] -= h
                    H[i, i] = (self(mu_p) - 2 * G0 + self(mu_m)) / h**2
                else:
                    mu_pp = mu.copy()
                    mu_pp[i] += h
                    mu_pp[j] += h
                    mu_pm = mu.copy()
                    mu_pm[i] += h
                    mu_pm[j] -= h
                    mu_mp = mu.copy()
                    mu_mp[i] -= h
                    mu_mp[j] += h
                    mu_mm = mu.copy()
                    mu_mm[i] -= h
                    mu_mm[j] -= h
                    H[i, j] = (
                        self(mu_pp) - self(mu_pm) - self(mu_mp) + self(mu_mm)
                    ) / (4 * h**2)
                    H[j, i] = H[i, j]
        return H


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def fernholz_weights(
    log_gradient: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute FGP weights from log-gradient of G and market weights.

    .. math::
        \pi_i = \left[D_i \log G(\mu)
                 + 1 - \sum_k \mu_k D_k \log G(\mu)\right] \mu_i

    Parameters
    ----------
    log_gradient : ndarray of shape (n,)
        [D_1 log G(μ), ..., D_n log G(μ)].
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights.

    References
    ----------
    F&K Survey Eq. 11.1
    """
    S = float(np.dot(mu, log_gradient))
    pi = (log_gradient + 1.0 - S) * mu
    return pi


def drift_process(
    G: GeneratingFunction,
    mu: NDArray[np.float64],
    tau_mu: NDArray[np.float64],
) -> float:
    r"""Compute drift g(t) of an FGP from the master formula.

    .. math::
        g(t) = -\frac{1}{2 G(\mu)} \sum_{i,j}
               D^2_{ij} G(\mu)\,\mu_i\,\mu_j\,\tau^{\mu}_{ij}

    Positive drift means the FGP outperforms the market beyond
    the boundary term log(G(μ(T))/G(μ(0))).

    Parameters
    ----------
    G : GeneratingFunction
        The generating function.
    mu : ndarray of shape (n,)
        Market weights.
    tau_mu : ndarray of shape (n, n)
        Relative covariance matrix of the market portfolio.

    Returns
    -------
    float
        Drift process value at this instant.

    References
    ----------
    F&K Survey Eq. 11.3
    """
    G_val = G(mu)
    require(G_val > 0, f"G(μ) must be positive, got {G_val}")
    H = G.hessian(mu)
    mu_outer = np.outer(mu, mu)
    return -0.5 / G_val * float(np.sum(H * tau_mu * mu_outer))
