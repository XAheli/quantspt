"""Rolling sample covariance estimation with annualisation.

Estimates the instantaneous covariance rate matrix a_{ij}(t) from
observed return data using rolling windows of log-returns.

Mathematical References
-----------------------
- Covariance rate a_{ij}: F&K Survey Eq. 1.3
- Annualisation convention: a_{ij} = Cov(dY_i, dY_j)/dt, with dt = 1/252
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._preconditions import require

__all__ = [
    "rolling_sample_covariance",
    "sample_covariance",
]


def sample_covariance(
    returns: NDArray[np.float64],
    *,
    annualize: bool = True,
    frequency: int = 252,
    min_observations: int | None = None,
) -> dict[str, NDArray[np.float64] | int | None]:
    r"""Compute sample covariance matrix from return data.

    Given T observations of n-dimensional returns, compute the sample
    covariance and optionally annualise by multiplying by the trading
    frequency (convention: a_{ij} × dt yields per-period covariance).

    Parameters
    ----------
    returns : ndarray of shape (T, n)
        Matrix of returns (log-returns recommended for SPT applications).
    annualize : bool
        If ``True`` (default), multiply by *frequency* to convert
        per-period covariance to annualised covariance rate.
    frequency : int
        Trading days per year, default 252.
    min_observations : int or None
        Minimum number of observations required. Defaults to ``n + 1``
        where n is the number of assets.

    Returns
    -------
    dict with keys:
        ``'raw'`` : ndarray of shape (n, n)
            Un-annualised sample covariance.
        ``'annualized'`` : ndarray of shape (n, n) or None
            Annualised covariance rate (raw × frequency).
            ``None`` if ``annualize=False``.
        ``'n_observations'`` : int
            Number of observations used.

    Raises
    ------
    SPTInvariantError
        If the number of observations is below ``min_observations``.

    References
    ----------
    F&K Survey Eq. 1.3: a_{ij}(t) dt = Cov(d log X_i, d log X_j)
    """
    returns = np.asarray(returns, dtype=np.float64)
    require(returns.ndim == 2, f"returns must be 2-D, got shape {returns.shape}")

    T, n = returns.shape
    if min_observations is None:
        min_observations = n + 1

    require(
        min_observations <= T,
        f"Need at least {min_observations} observations, got {T}",
    )

    raw: NDArray[np.float64] = np.asarray(
        np.cov(returns, rowvar=False, ddof=1), dtype=np.float64
    )
    if raw.ndim == 0:
        raw = raw.reshape(1, 1)

    annualized: NDArray[np.float64] | None = (
        np.asarray(raw * frequency, dtype=np.float64) if annualize else None
    )
    result: dict[str, NDArray[np.float64] | int | None] = {
        "raw": raw,
        "annualized": annualized,
        "n_observations": T,
    }
    return result


def rolling_sample_covariance(
    returns: NDArray[np.float64],
    window: int,
    *,
    annualize: bool = True,
    frequency: int = 252,
    min_periods: int | None = None,
) -> list[dict[str, NDArray[np.float64] | int | None]]:
    r"""Compute rolling-window sample covariance matrices.

    For each time step t = window−1, ..., T−1, compute the sample
    covariance using observations [t−window+1, ..., t].

    Parameters
    ----------
    returns : ndarray of shape (T, n)
        Matrix of returns.
    window : int
        Rolling window size (number of observations per estimate).
    annualize : bool
        If ``True``, multiply by *frequency*.
    frequency : int
        Trading days per year.
    min_periods : int or None
        Minimum valid observations in a window.  Defaults to ``n + 1``.

    Returns
    -------
    list of dict
        One dict per time step, same structure as :func:`sample_covariance`.
        Length is ``T − window + 1``.

    References
    ----------
    F&K Survey Eq. 1.3
    """
    returns = np.asarray(returns, dtype=np.float64)
    require(returns.ndim == 2, f"returns must be 2-D, got shape {returns.shape}")

    T, n = returns.shape
    if min_periods is None:
        min_periods = n + 1

    require(window >= min_periods, f"window ({window}) < min_periods ({min_periods})")
    require(window <= T, f"Need at least {window} observations, got {T}")

    results: list[dict[str, NDArray[np.float64] | int | None]] = []
    for t in range(window - 1, T):
        chunk = returns[t - window + 1 : t + 1]
        results.append(
            sample_covariance(
                chunk,
                annualize=annualize,
                frequency=frequency,
                min_observations=min_periods,
            )
        )
    return results
