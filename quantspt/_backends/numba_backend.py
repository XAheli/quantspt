"""Numba compute backend — JIT-compiled inner loops.

Provides @numba.njit compiled versions of tight numerical loops
for covariance computation and path simulation. Falls back gracefully
to the NumPy backend if numba is not installed.

Requires the ``sim`` extra: ``pip install quantspt[sim]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["NumbaBackend"]


def _require_numba() -> Any:
    """Import numba or raise with installation instructions."""
    try:
        import numba

        return numba
    except ImportError as exc:
        raise ImportError(
            "Numba is required for the Numba backend. "
            "Install with: pip install quantspt[sim]"
        ) from exc


def _build_numba_functions() -> dict[str, Any]:
    """Build JIT-compiled functions using numba."""
    numba = _require_numba()
    njit = numba.njit

    @njit(cache=True)  # type: ignore[misc, untyped-decorator]
    def _excess_growth_rate(pi: Any, a_diag: Any, a: Any) -> Any:
        weighted_var = 0.0
        n = len(pi)
        for i in range(n):
            weighted_var += pi[i] * a_diag[i]

        port_var = 0.0
        for i in range(n):
            for j in range(n):
                port_var += pi[i] * a[i, j] * pi[j]

        return 0.5 * (weighted_var - port_var)

    @njit(cache=True)  # type: ignore[misc, untyped-decorator]
    def _relative_covariance(a: Any, pi: Any) -> Any:
        n = len(pi)
        a_pi = np.zeros(n)
        for i in range(n):
            for j in range(n):
                a_pi[i] += a[i, j] * pi[j]

        a_pipi = 0.0
        for i in range(n):
            a_pipi += pi[i] * a_pi[i]

        tau = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                tau[i, j] = a[i, j] - a_pi[i] - a_pi[j] + a_pipi
        return tau

    @njit(cache=True)  # type: ignore[misc, untyped-decorator]
    def _simulate_gbm_paths(
        x0: Any,
        mu: Any,
        cholesky: Any,
        dt: Any,
        n_steps: Any,
        random_increments: Any,
    ) -> Any:
        n_assets = len(x0)
        paths = np.empty((n_steps + 1, n_assets))
        paths[0] = x0

        a_diag = np.zeros(n_assets)
        for i in range(n_assets):
            for k in range(n_assets):
                a_diag[i] += cholesky[i, k] ** 2

        for t in range(n_steps):
            dw = random_increments[t]
            for i in range(n_assets):
                log_inc = (mu[i] - 0.5 * a_diag[i]) * dt
                for k in range(n_assets):
                    log_inc += cholesky[i, k] * dw[k]
                paths[t + 1, i] = paths[t, i] * np.exp(log_inc)

        return paths

    @njit(cache=True)  # type: ignore[misc, untyped-decorator]
    def _covariance_inner(returns: Any, n_obs: Any, n_assets: Any) -> Any:
        mean = np.zeros(n_assets)
        for i in range(n_assets):
            s = 0.0
            for t in range(n_obs):
                s += returns[t, i]
            mean[i] = s / n_obs

        cov = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(i, n_assets):
                s = 0.0
                for t in range(n_obs):
                    s += (returns[t, i] - mean[i]) * (returns[t, j] - mean[j])
                cov[i, j] = s / (n_obs - 1)
                cov[j, i] = cov[i, j]
        return cov

    return {
        "excess_growth_rate": _excess_growth_rate,
        "relative_covariance": _relative_covariance,
        "simulate_gbm_paths": _simulate_gbm_paths,
        "covariance_inner": _covariance_inner,
    }


class NumbaBackend:
    """Numba JIT-compiled backend for tight inner loops.

    Operations are compiled on first invocation and cached for
    subsequent calls. Particularly effective for path simulation
    and covariance computation with many assets.
    """

    def __init__(self) -> None:
        self._fns = _build_numba_functions()

    @property
    def name(self) -> str:
        return "numba"

    def excess_growth_rate(
        self,
        pi: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> float:
        r"""Numba-compiled excess growth rate γ*_π."""
        return float(
            self._fns["excess_growth_rate"](
                np.ascontiguousarray(pi),
                np.ascontiguousarray(np.diag(a)),
                np.ascontiguousarray(a),
            )
        )

    def relative_covariance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Numba-compiled relative covariance τ^π."""
        result: NDArray[np.float64] = self._fns["relative_covariance"](
            np.ascontiguousarray(a),
            np.ascontiguousarray(pi),
        )
        return result

    def portfolio_variance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> float:
        """Portfolio variance via numba (delegates to matrix multiply)."""
        return float(pi @ a @ pi)

    def simulate_gbm_step(
        self,
        x: NDArray[np.float64],
        mu: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Single GBM step (not JIT — use simulate_gbm_paths for bulk)."""
        a_diag = np.sum(cholesky**2, axis=1)
        log_increment = (mu - 0.5 * a_diag) * dt + cholesky @ dw
        result: NDArray[np.float64] = x * np.exp(log_increment)
        return result

    def simulate_gbm_paths(
        self,
        x0: NDArray[np.float64],
        mu: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        dt: float,
        n_steps: int,
        random_increments: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Simulate full GBM paths using Numba JIT inner loop.

        Parameters
        ----------
        x0 : ndarray of shape (n,)
            Initial prices.
        mu : ndarray of shape (n,)
            Drift vector.
        cholesky : ndarray of shape (n, n)
            Cholesky factor of covariance matrix.
        dt : float
            Time step size.
        n_steps : int
            Number of steps to simulate.
        random_increments : ndarray of shape (n_steps, n)
            Pre-generated N(0, sqrt(dt)) increments.

        Returns
        -------
        ndarray of shape (n_steps + 1, n)
            Simulated price paths.
        """
        result: NDArray[np.float64] = self._fns["simulate_gbm_paths"](
            np.ascontiguousarray(x0),
            np.ascontiguousarray(mu),
            np.ascontiguousarray(cholesky),
            dt,
            n_steps,
            np.ascontiguousarray(random_increments),
        )
        return result

    def diversity_weights(
        self,
        mu: NDArray[np.float64],
        p: float,
    ) -> NDArray[np.float64]:
        """Diversity-weighted portfolio."""
        mu_p = mu**p
        result: NDArray[np.float64] = mu_p / np.sum(mu_p)
        return result

    def covariance_shrinkage(
        self,
        returns: NDArray[np.float64],
        shrinkage: float,
    ) -> NDArray[np.float64]:
        """Shrinkage covariance with Numba-accelerated sample cov."""
        n_obs, n_assets = returns.shape
        sample_cov: NDArray[np.float64] = self._fns["covariance_inner"](
            np.ascontiguousarray(returns),
            n_obs,
            n_assets,
        )
        target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        result: NDArray[np.float64] = (
            1.0 - shrinkage
        ) * sample_cov + shrinkage * target
        return result
