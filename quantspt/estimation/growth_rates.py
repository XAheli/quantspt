r"""Growth rate estimation from return data.

Estimates individual stock growth rates γ_i from observed log-returns.
In SPT, the growth rate of stock i is:

.. math::
    \gamma_i = b_i - \frac{a_{ii}}{2}

where b_i is the drift rate and a_{ii} is the variance rate. For
log-returns r_t = log(X_t/X_{t-1}), the growth rate equals E[r_t]/dt
in the continuous-time limit.

Mathematical References
-----------------------
- Growth rate definition: F&K Survey Eq. 1.4-1.6
- Portfolio growth rate decomposition: F&K Survey Eq. 1.12
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "estimate_growth_rates",
    "rolling_growth_rates",
]


def estimate_growth_rates(
    log_returns: NDArray[np.float64],
    *,
    frequency: int = 252,
    bias_correction: bool = True,
    min_observations: int | None = None,
) -> dict[str, NDArray[np.float64] | int]:
    r"""Estimate individual stock growth rates from log-return data.

    The growth rate is estimated as:

    .. math::
        \hat{\gamma}_i = \bar{r}_i \times f

    where :math:`\bar{r}_i` is the sample mean of log-returns for stock i
    and f is the annualisation frequency. With bias correction, we apply
    the finite-sample adjustment:

    .. math::
        \hat{\gamma}^{\text{bc}}_i = \hat{\gamma}_i
            + \frac{\hat{a}_{ii}}{2}
            \cdot \frac{1}{T}

    which corrects for Jensen's inequality bias in finite samples.

    Parameters
    ----------
    log_returns : ndarray of shape (T, n)
        Matrix of log-returns (r_t = log(X_t / X_{t-1})).
    frequency : int
        Annualisation factor (252 for daily, 52 for weekly, 12 for monthly).
    bias_correction : bool
        If ``True`` (default), apply finite-sample bias correction.
    min_observations : int or None
        Minimum required observations. Defaults to 10.

    Returns
    -------
    dict with keys:
        ``'growth_rates'`` : ndarray of shape (n,)
            Annualised growth rate estimates γ_i.
        ``'standard_errors'`` : ndarray of shape (n,)
            Standard errors of the growth rate estimates.
        ``'n_observations'`` : int
            Number of observations used.

    References
    ----------
    F&K Survey Eq. 1.4-1.6
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    require(
        log_returns.ndim == 2,
        f"log_returns must be 2-D, got shape {log_returns.shape}",
    )

    T, n = log_returns.shape
    if min_observations is None:
        min_observations = 10
    require(
        min_observations <= T,
        f"Need at least {min_observations} observations, got {T}",
    )

    mean_returns = np.mean(log_returns, axis=0)
    gamma: NDArray[np.float64] = mean_returns * frequency

    if bias_correction:
        var_returns = np.var(log_returns, axis=0, ddof=1)
        gamma = gamma + 0.5 * var_returns * frequency / T

    se = np.std(log_returns, axis=0, ddof=1) * np.sqrt(frequency / T)

    return {
        "growth_rates": gamma,
        "standard_errors": se,
        "n_observations": T,
    }


def rolling_growth_rates(
    log_returns: NDArray[np.float64],
    window: int,
    *,
    frequency: int = 252,
    bias_correction: bool = True,
) -> list[dict[str, NDArray[np.float64] | int]]:
    r"""Rolling-window growth rate estimation.

    Parameters
    ----------
    log_returns : ndarray of shape (T, n)
        Matrix of log-returns.
    window : int
        Rolling window size.
    frequency : int
        Annualisation factor.
    bias_correction : bool
        Apply finite-sample bias correction.

    Returns
    -------
    list of dict
        One result dict per time step, length ``T - window + 1``.

    References
    ----------
    F&K Survey Eq. 1.4-1.6
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    require(
        log_returns.ndim == 2,
        f"log_returns must be 2-D, got shape {log_returns.shape}",
    )

    T = log_returns.shape[0]
    require(window <= T, f"Need at least {window} observations, got {T}")

    results: list[dict[str, NDArray[np.float64] | int]] = []
    for t in range(window - 1, T):
        chunk = log_returns[t - window + 1 : t + 1]
        results.append(
            estimate_growth_rates(
                chunk,
                frequency=frequency,
                bias_correction=bias_correction,
                min_observations=2,
            )
        )
    return results
