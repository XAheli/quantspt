"""Master formula for Functionally Generated Portfolios.

The master formula is the key decomposition theorem of SPT:

    log(V^π(T) / V^μ(T)) = log(G(μ(T)) / G(μ(0))) + ∫₀ᵀ g(t) dt

It completely characterizes FGP performance as a boundary term (observable
from market weights alone) plus a drift integral.

Mathematical References
-----------------------
- Master formula: F&K Survey Eq. 11.2
- Drift process: F&K Survey Eq. 11.3
- Verification methodology: compare simulated relative returns against
  the boundary + drift decomposition
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .covariance import relative_covariance
from .generating_functions import GeneratingFunction

__all__ = [
    "boundary_term",
    "drift_integral",
    "master_formula_decomposition",
    "verify_master_formula",
]


def boundary_term(
    G: GeneratingFunction,
    mu_T: NDArray[np.float64],
    mu_0: NDArray[np.float64],
) -> float:
    r"""Compute the boundary term of the master formula.

    .. math::
        \text{boundary} = \log\frac{G(\mu(T))}{G(\mu(0))}

    This term depends only on market weights at start and end, not on
    the path. It can be positive or negative.

    Parameters
    ----------
    G : GeneratingFunction
        Portfolio generating function.
    mu_T : ndarray of shape (n,)
        Market weights at time T.
    mu_0 : ndarray of shape (n,)
        Market weights at time 0.

    Returns
    -------
    float
        Log-ratio of G evaluated at terminal vs initial weights.

    References
    ----------
    F&K Survey Eq. 11.2 (first term on RHS)
    """
    G_T = G(mu_T)
    G_0 = G(mu_0)
    require(G_0 > 0, "G(μ(0)) must be positive")
    require(G_T > 0, "G(μ(T)) must be positive")
    return float(np.log(G_T / G_0))


def drift_integral(
    G: GeneratingFunction,
    mu_path: NDArray[np.float64],
    a_path: NDArray[np.float64],
    dt: float,
) -> float:
    r"""Compute the drift integral ∫₀ᵀ g(t) dt via left Riemann sum.

    .. math::
        \int_0^T g(t)\,dt \approx \sum_{k=0}^{N-1} g(t_k) \Delta t

    where g(t) is the drift process from the master formula and N is the
    number of intervals (= len(mu_path) - 1).

    Parameters
    ----------
    G : GeneratingFunction
        Portfolio generating function.
    mu_path : ndarray of shape (N+1, n)
        Time series of market weights at N+1 time points (N intervals).
    a_path : ndarray of shape (N+1, n, n)
        Time series of covariance rate matrices.
    dt : float
        Time step between observations (in years).

    Returns
    -------
    float
        Approximate drift integral.

    References
    ----------
    F&K Survey Eq. 11.2 (second term on RHS), Eq. 11.3 (integrand)
    """
    T_steps = len(mu_path)
    require(T_steps >= 2, "Need at least 2 time steps")
    require(a_path.shape[0] == T_steps, "mu_path and a_path must have same length")

    total = 0.0
    for t in range(T_steps - 1):
        mu_t = mu_path[t]
        a_t = a_path[t]
        tau_mu_t = relative_covariance(a_t, mu_t)
        g_t = G.drift(mu_t, tau_mu_t)
        total += g_t

    return total * dt


def master_formula_decomposition(
    G: GeneratingFunction,
    mu_path: NDArray[np.float64],
    a_path: NDArray[np.float64],
    dt: float,
) -> dict[str, float]:
    r"""Full master formula decomposition of FGP performance.

    Returns the boundary term, drift integral, and their sum which
    should equal the log-relative return of the FGP.

    .. math::
        \underbrace{\log\frac{V^{\pi}(T)}{V^{\mu}(T)}}_{\text{total}}
        = \underbrace{\log\frac{G(\mu(T))}{G(\mu(0))}}_{\text{boundary}}
        + \underbrace{\int_0^T g(t)\,dt}_{\text{drift}}

    Parameters
    ----------
    G : GeneratingFunction
        Portfolio generating function.
    mu_path : ndarray of shape (T, n)
        Time series of market weights.
    a_path : ndarray of shape (T, n, n)
        Time series of covariance rate matrices.
    dt : float
        Time step between observations.

    Returns
    -------
    dict with keys:
        - 'boundary': log(G(μ_T)/G(μ_0))
        - 'drift_integral': ∫g(t)dt
        - 'total': boundary + drift_integral

    References
    ----------
    F&K Survey Eq. 11.2
    """
    bnd = boundary_term(G, mu_path[-1], mu_path[0])
    drift = drift_integral(G, mu_path, a_path, dt)
    return {
        "boundary": bnd,
        "drift_integral": drift,
        "total": bnd + drift,
    }


def verify_master_formula(
    G: GeneratingFunction,
    mu_path: NDArray[np.float64],
    a_path: NDArray[np.float64],
    log_relative_return: float,
    dt: float,
    rtol: float = 0.05,
) -> dict[str, object]:
    r"""Verify that the master formula holds for simulated/empirical data.

    Checks whether the actual log-relative return matches the
    boundary + drift decomposition within tolerance.

    Parameters
    ----------
    G : GeneratingFunction
        Portfolio generating function.
    mu_path : ndarray of shape (T, n)
        Time series of market weights.
    a_path : ndarray of shape (T, n, n)
        Time series of covariance rate matrices.
    log_relative_return : float
        Actual observed log(V^π(T) / V^μ(T)).
    dt : float
        Time step.
    rtol : float
        Relative tolerance for verification.

    Returns
    -------
    dict with keys:
        - 'verified': bool
        - 'actual': observed log-relative return
        - 'predicted': boundary + drift
        - 'boundary': boundary term
        - 'drift_integral': drift term
        - 'error': actual - predicted
        - 'relative_error': |error| / |actual| (or absolute if actual ≈ 0)
    """
    decomp = master_formula_decomposition(G, mu_path, a_path, dt)
    predicted = decomp["total"]
    error = log_relative_return - predicted

    if abs(log_relative_return) > 1e-10:
        rel_error = abs(error / log_relative_return)
    else:
        rel_error = abs(error)

    return {
        "verified": rel_error <= rtol,
        "actual": log_relative_return,
        "predicted": predicted,
        "boundary": decomp["boundary"],
        "drift_integral": decomp["drift_integral"],
        "error": error,
        "relative_error": rel_error,
    }
