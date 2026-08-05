r"""Atlas model calibration from market data.

Fits Atlas/first-order model parameters to observed capital distribution
data using maximum likelihood estimation of rank-dependent growth rates
and volatilities.

Calibration procedure (BFK Sec. 6.4-6.5):
1. Compute ranked market weights from capitalisation data
2. Estimate rank-dependent volatilities from ranked return variances
3. Estimate Pareto exponents from ranked weight ratios via MLE
4. Recover growth parameters g_k from Pareto exponents and volatilities
5. Validate the stability condition (BFK Eq. 1.5)

Mathematical References
-----------------------
- Atlas dynamics: BFK Eq. 1.1, 1.6-1.7
- Stability condition: BFK Eq. 1.5
- Pareto exponents: BFK Eq. 4.3-4.4
- Capital distribution curve: BFK Eq. 4.12-4.15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .._preconditions import require
from ..errors import CalibrationError

if TYPE_CHECKING:
    pass

__all__ = [
    "calibrate_atlas",
    "goodness_of_fit",
]


def calibrate_atlas(
    market_caps: NDArray[np.float64],
    *,
    dt: float = 1.0 / 252,
    min_observations: int = 50,
) -> dict[str, NDArray[np.float64] | float | int]:
    r"""Calibrate a first-order Atlas model to market capitalisation data.

    Given a time series of market capitalisations, estimate rank-dependent
    growth and volatility parameters that best fit the observed capital
    distribution dynamics.

    Parameters
    ----------
    market_caps : ndarray of shape (T, n)
        Time series of market capitalisations (positive values).
    dt : float
        Time step between observations (default 1/252 for daily).
    min_observations : int
        Minimum number of time steps required.

    Returns
    -------
    dict with keys:
        ``'n'`` : int
            Number of stocks.
        ``'gamma'`` : float
            Common drift parameter.
        ``'g'`` : ndarray of shape (n,)
            Rank-dependent growth-rate increments.
        ``'sigma'`` : ndarray of shape (n,)
            Rank-dependent volatilities.
        ``'pareto_exponents'`` : ndarray of shape (n-1,)
            Estimated Pareto exponents r_k.
        ``'n_observations'`` : int
            Number of time steps used.

    Raises
    ------
    CalibrationError
        If the stability condition cannot be satisfied.

    References
    ----------
    BFK Eq. 1.5-1.7, 4.3-4.4, 6.4-6.5
    """
    market_caps = np.asarray(market_caps, dtype=np.float64)
    require(
        market_caps.ndim == 2,
        f"market_caps must be 2-D, got shape {market_caps.shape}",
    )

    T, n = market_caps.shape
    require(
        min_observations <= T,
        f"Need at least {min_observations} observations, got {T}",
    )
    require(n >= 2, f"Need at least 2 stocks, got {n}")
    require(
        bool(np.all(market_caps > 0)),
        "All market capitalisations must be positive",
    )

    weights = market_caps / market_caps.sum(axis=1, keepdims=True)

    ranked_weights = np.sort(weights, axis=1)[:, ::-1]

    log_caps = np.log(market_caps)
    ranked_indices = np.argsort(-market_caps, axis=1)

    sigma_sq = np.zeros(n)
    for k in range(n):
        rank_k_returns = []
        for t in range(1, T):
            stock_at_rank_k_prev = ranked_indices[t - 1, k]
            stock_at_rank_k_curr = ranked_indices[t, k]
            if stock_at_rank_k_prev == stock_at_rank_k_curr:
                ret = (
                    log_caps[t, stock_at_rank_k_curr]
                    - log_caps[t - 1, stock_at_rank_k_prev]
                )
                rank_k_returns.append(ret)
        if len(rank_k_returns) > 10:
            sigma_sq[k] = float(np.var(rank_k_returns, ddof=1) / dt)
        else:
            sigma_sq[k] = float(np.var(np.diff(log_caps[:, 0]), ddof=1) / dt)

    sigma = np.sqrt(np.maximum(sigma_sq, 1e-10))

    pareto_exp_arr = np.zeros(n - 1)
    for k in range(n - 1):
        ratios = ranked_weights[:, k] / np.maximum(ranked_weights[:, k + 1], 1e-15)
        valid_ratios = ratios[ratios > 1.0]
        if len(valid_ratios) > 10:
            log_ratios = np.log(valid_ratios)
            pareto_exp_arr[k] = 1.0 / float(np.mean(log_ratios))
        else:
            pareto_exp_arr[k] = 1.0

    pareto_exp: NDArray[np.float64] = np.asarray(
        np.maximum(pareto_exp_arr, 0.01), dtype=np.float64
    )

    g: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    cumsum_g = np.zeros(n)
    for k in range(n - 1):
        cumsum_g[k] = -pareto_exp[k] * (sigma_sq[k] + sigma_sq[k + 1]) / 4.0

    for k in range(n - 1):
        if k == 0:
            g[k] = cumsum_g[k]
        else:
            g[k] = cumsum_g[k] - cumsum_g[k - 1]

    g[n - 1] = -np.sum(g[: n - 1])

    if not np.isclose(np.sum(g), 0.0, atol=1e-8):
        raise CalibrationError(
            f"Stability condition violated: g sums to {np.sum(g):.2e}, not 0"
        )

    cumsum_check = np.cumsum(g)
    if not np.all(cumsum_check[:-1] < 0):
        g = np.asarray(_enforce_stability(g, sigma_sq, pareto_exp), dtype=np.float64)

    log_returns = np.diff(np.log(market_caps.sum(axis=1)))
    gamma = float(np.mean(log_returns) / dt)

    return {
        "n": n,
        "gamma": gamma,
        "g": g,
        "sigma": sigma,
        "pareto_exponents": pareto_exp,
        "n_observations": T,
    }


def _enforce_stability(
    g: NDArray[np.float64],
    sigma_sq: NDArray[np.float64],
    pareto_exp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Adjust g to satisfy the stability condition (BFK Eq. 1.5).

    Ensures partial sums g_1 + ... + g_k < 0 for k = 1,...,n-1
    and total sum = 0.
    """
    n = len(g)
    g_new = g.copy()

    cumsum = np.cumsum(g_new)
    for k in range(n - 1):
        if cumsum[k] >= 0:
            adjustment = cumsum[k] + 0.001 * np.mean(sigma_sq)
            g_new[k] -= adjustment
            cumsum = np.cumsum(g_new)

    g_new[n - 1] = -np.sum(g_new[: n - 1])
    return g_new


def goodness_of_fit(
    market_caps: NDArray[np.float64],
    calibrated_params: dict[str, NDArray[np.float64] | float | int],
    *,
    dt: float = 1.0 / 252,
) -> dict[str, NDArray[np.float64] | float]:
    r"""Assess the quality of an Atlas model calibration.

    Returns diagnostics comparing the fitted model's predictions against
    observed market data.

    Parameters
    ----------
    market_caps : ndarray of shape (T, n)
        Original market capitalisation data.
    calibrated_params : dict
        Output from :func:`calibrate_atlas`.
    dt : float
        Time step between observations.

    Returns
    -------
    dict with keys:
        ``'ks_pvalues'`` : ndarray of shape (n-1,)
            Kolmogorov-Smirnov test p-values for Pareto fit at each rank.
        ``'capital_curve_rmse'`` : float
            RMSE between fitted and observed ranked weight curves.
        ``'ergodic_deviation'`` : ndarray of shape (n,)
            Deviation of time-at-rank from the ergodic prediction 1/n.

    References
    ----------
    BFK Sec. 6.4-6.5
    """
    market_caps = np.asarray(market_caps, dtype=np.float64)
    T, n = market_caps.shape

    weights = market_caps / market_caps.sum(axis=1, keepdims=True)
    ranked_weights = np.sort(weights, axis=1)[:, ::-1]

    pareto_exp = np.asarray(calibrated_params["pareto_exponents"])
    ks_pvalues = np.zeros(n - 1)
    for k in range(n - 1):
        ratios = ranked_weights[:, k] / np.maximum(ranked_weights[:, k + 1], 1e-15)
        valid_ratios = ratios[ratios > 1.0]
        if len(valid_ratios) > 10:
            log_ratios = np.log(valid_ratios)
            fitted_rate = pareto_exp[k]
            _, pval = stats.kstest(log_ratios, "expon", args=(0, 1.0 / fitted_rate))
            ks_pvalues[k] = pval
        else:
            ks_pvalues[k] = np.nan

    from ..models.atlas import FirstOrderModel

    params = calibrated_params
    model = FirstOrderModel(
        n=int(params["n"]),
        gamma=float(params["gamma"]),
        g=np.asarray(params["g"]),
        sigma=np.asarray(params["sigma"]),
    )
    ce_weights = model.certainty_equivalent_weights()
    observed_mean_weights = np.mean(ranked_weights, axis=0)
    capital_curve_rmse = float(
        np.sqrt(np.mean((ce_weights - observed_mean_weights) ** 2))
    )

    ranked_indices = np.argsort(-market_caps, axis=1)
    rank_counts = np.zeros((n, n))
    for t in range(T):
        for stock_idx in range(n):
            rank = ranked_indices[t, stock_idx]
            rank_counts[stock_idx, rank] += 1
    time_at_rank = rank_counts / T
    ergodic_prediction = 1.0 / n
    ergodic_deviation = np.mean(np.abs(time_at_rank - ergodic_prediction), axis=0)

    return {
        "ks_pvalues": ks_pvalues,
        "capital_curve_rmse": capital_curve_rmse,
        "ergodic_deviation": ergodic_deviation,
    }
