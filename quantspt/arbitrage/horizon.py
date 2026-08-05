"""Arbitrage horizon analysis.

Computes the minimum time horizon required for relative arbitrage
under various conditions, and analyses the sensitivity of the horizon
to model parameters.

Mathematical References
-----------------------
- Diversity-based horizon: FKK Eq. 4.5
- Entropy-based horizon: F&K Survey Eq. 11.8–11.12
- Horizon sensitivity: FKK §4
"""

from __future__ import annotations

import numpy as np

from .._preconditions import require

__all__ = [
    "diversity_horizon",
    "entropy_horizon",
    "horizon_sensitivity",
]


def diversity_horizon(
    n: int,
    p: float,
    eps: float,
    delta: float,
) -> float:
    r"""Minimum horizon for diversity-based relative arbitrage.

    Under the weak diversity condition with parameters (δ, p) and
    non-degeneracy constant ε:

    .. math::
        T^* = \frac{2 \log n}{p\,\varepsilon\,\delta}

    For T ≥ T*, the diversity-weighted portfolio outperforms the
    market almost surely.

    Parameters
    ----------
    n : int
        Number of stocks (≥ 2).
    p : float
        Diversity exponent p ∈ (0, 1).
    eps : float
        Non-degeneracy constant ε > 0.
    delta : float
        Diversity parameter δ > 0.

    Returns
    -------
    float
        Minimum horizon T*.

    References
    ----------
    FKK Eq. 4.5
    """
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(eps > 0, f"ε must be positive, got {eps}")
    require(delta > 0, f"δ must be positive, got {delta}")
    return float(2.0 * np.log(n) / (p * eps * delta))


def entropy_horizon(
    H_mu_0: float,
    zeta: float,
) -> float:
    r"""Minimum horizon for entropy-based relative arbitrage.

    If the excess growth rate of the market portfolio satisfies
    γ*_μ(t) ≥ ζ > 0 for all t, then relative arbitrage exists for:

    .. math::
        T > \frac{H(\mu(0))}{\zeta}

    where H(μ(0)) = −Σ μ_i(0) log μ_i(0) is the initial entropy of
    market weights.

    Parameters
    ----------
    H_mu_0 : float
        Shannon entropy of initial market weights.
    zeta : float
        Lower bound on excess growth rate ζ > 0.

    Returns
    -------
    float
        Minimum horizon.

    References
    ----------
    F&K Survey Eq. 11.8–11.12
    """
    require(H_mu_0 >= 0, f"Entropy must be non-negative, got {H_mu_0}")
    require(zeta > 0, f"ζ must be positive, got {zeta}")
    return H_mu_0 / zeta


def horizon_sensitivity(
    n: int,
    p: float,
    eps: float,
    delta: float,
    perturbation: float = 0.01,
) -> dict[str, float]:
    r"""Sensitivity of the arbitrage horizon to each parameter.

    Computes the partial derivative of T* = 2 log(n)/(pεδ) with respect
    to each parameter, expressed as semi-elasticities (% change in T*
    per unit change in parameter).

    Parameters
    ----------
    n : int
        Number of stocks (≥ 2).
    p : float
        Diversity exponent p ∈ (0, 1).
    eps : float
        Non-degeneracy constant ε > 0.
    delta : float
        Diversity parameter δ > 0.
    perturbation : float
        Relative perturbation size for numerical derivatives.

    Returns
    -------
    dict with keys ``'n'``, ``'p'``, ``'eps'``, ``'delta'``
        Semi-elasticity of T* with respect to each parameter.
        Negative values mean increasing the parameter reduces the
        horizon (makes arbitrage faster).

    References
    ----------
    FKK §4
    """
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(eps > 0, f"ε must be positive, got {eps}")
    require(delta > 0, f"δ must be positive, got {delta}")
    require(perturbation > 0, f"perturbation must be positive, got {perturbation}")

    T0 = diversity_horizon(n, p, eps, delta)

    dp = perturbation
    T_p_up = diversity_horizon(n, min(p + dp, 0.999), eps, delta)
    T_p_dn = diversity_horizon(n, max(p - dp, 0.001), eps, delta)
    sens_p = (T_p_up - T_p_dn) / (min(p + dp, 0.999) - max(p - dp, 0.001)) / T0

    de = eps * perturbation
    sens_eps = (diversity_horizon(n, p, eps + de, delta) - T0) / de / T0

    dd = delta * perturbation
    sens_delta = (diversity_horizon(n, p, eps, delta + dd) - T0) / dd / T0

    n_up = n + 1
    sens_n = (diversity_horizon(n_up, p, eps, delta) - T0) / T0

    return {
        "n": sens_n,
        "p": sens_p,
        "eps": sens_eps,
        "delta": sens_delta,
    }
