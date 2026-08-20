"""DCC-GARCH dynamic conditional correlation model.

Produces a time-varying covariance matrix Σ(t) at each observation, making
it possible to track how correlations evolve through crises and calm periods.
Implements the ``CovarianceRateProcess`` protocol so the output plugs directly
into quantspt's γ* computation pipeline.

Mathematical References
-----------------------
- DCC model: Engle (2002), "Dynamic Conditional Correlation: A Simple Class
  of Multivariate Generalized Autoregressive Conditional Heteroskedasticity
  Models," Journal of Business & Economic Statistics 20(3), pp. 339-350.

  Two-step estimation:
    Step 1 — fit univariate GARCH(1,1) to each asset:
      σ²_{i,t} = ω_i + α_i ε²_{i,t-1} + β_i σ²_{i,t-1}

    Step 2 — model time-varying correlations on standardized residuals:
      Q_t = (1 − a − b) Q̄ + a ε_{t-1} ε'_{t-1} + b Q_{t-1}
      R_t = diag(Q_t)^{−1/2}  Q_t  diag(Q_t)^{−1/2}

  where Q̄ = (1/T) Σ_t ε_t ε_t' is the unconditional (uncentered) second
  moment of standardized residuals, and a, b are scalar DCC parameters
  with a + b < 1.

- Final covariance:
      Σ_t = D_t R_t D_t
  where D_t = diag(σ_{1,t}, …, σ_{n,t}).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from .._preconditions import require
from .._result import SPTResult

__all__ = [
    "DCCGarch",
    "DCCResult",
]


@dataclass(frozen=True)
class DCCResult:
    """Fitted DCC-GARCH model output.

    Attributes
    ----------
    covariances : NDArray[np.float64]
        Time-varying covariance matrices, shape ``(T, n, n)``.
    correlations : NDArray[np.float64]
        Time-varying correlation matrices, shape ``(T, n, n)``.
    conditional_vols : NDArray[np.float64]
        Conditional volatilities, shape ``(T, n)``.
    dcc_a : float
        DCC parameter a (news impact).
    dcc_b : float
        DCC parameter b (persistence).
    garch_params : list[dict[str, float]]
        Univariate GARCH(1,1) parameters per asset.
    """

    covariances: NDArray[np.float64]
    correlations: NDArray[np.float64]
    conditional_vols: NDArray[np.float64]
    dcc_a: float
    dcc_b: float
    garch_params: list[dict[str, float]]


class DCCGarch:
    r"""DCC-GARCH(1,1) estimator with CovarianceRateProcess interface.

    Two-step estimation following Engle (2002):

    **Step 1**: Fit GARCH(1,1) to each marginal return series to obtain
    conditional volatilities σ_{i,t} and standardized residuals
    ε_{i,t} = r_{i,t} / σ_{i,t}.

    **Step 2**: Estimate DCC parameters (a, b) on the standardized
    residuals via composite likelihood:

    .. math::
        Q_t = (1 - a - b)\,\bar{Q} + a\,\varepsilon_{t-1}
              \varepsilon_{t-1}' + b\,Q_{t-1}

    The correlation matrix at time t is:

    .. math::
        R_t = \operatorname{diag}(Q_t)^{-1/2}\,Q_t\,
              \operatorname{diag}(Q_t)^{-1/2}

    Parameters
    ----------
    use_arch : bool
        If True and the ``arch`` package is available, delegate GARCH
        fitting to ``arch.univariate.arch_model``. Otherwise use a
        simple internal estimator.

    References
    ----------
    Engle (2002), "Dynamic Conditional Correlation," JBES 20(3),
    pp. 339-350, Eq. (3)-(7).
    """

    def __init__(self, use_arch: bool = True) -> None:
        self._use_arch = use_arch
        self._fitted = False
        self._result: DCCResult | None = None
        self._times: NDArray[np.float64] | None = None

    def fit(
        self,
        returns: pd.DataFrame | NDArray[np.float64],
        *,
        times: NDArray[np.float64] | None = None,
    ) -> SPTResult[DCCResult]:
        """Fit DCC-GARCH to a panel of returns.

        Parameters
        ----------
        returns : DataFrame or ndarray of shape (T, n)
            Daily return series.
        times : ndarray of shape (T,), optional
            Time labels for each row (used by ``covariance_at``).
            Defaults to ``np.arange(T)``.

        Returns
        -------
        SPTResult[DCCResult]
            Fitted DCC model with time-varying covariance matrices.
        """
        t0 = time.perf_counter()

        arr: NDArray[np.float64]
        if isinstance(returns, pd.DataFrame):
            arr = returns.values.astype(np.float64)
        else:
            arr = np.asarray(returns, dtype=np.float64)
        require(arr.ndim == 2, f"returns must be 2-D, got {arr.ndim}-D")
        T, n = arr.shape
        require(T >= 10, f"Need at least 10 observations, got {T}")
        require(n >= 2, f"Need at least 2 assets, got {n}")

        if times is not None:
            self._times = np.asarray(times, dtype=np.float64)
        else:
            self._times = np.arange(T, dtype=np.float64)

        cond_vols = np.zeros((T, n), dtype=np.float64)
        garch_params: list[dict[str, float]] = []

        for i in range(n):
            omega, a_g, b_g, sigma2 = self._fit_garch(arr[:, i])
            cond_vols[:, i] = np.sqrt(sigma2)
            garch_params.append({"omega": omega, "alpha": a_g, "beta": b_g})

        safe_vols = np.where(cond_vols > 1e-15, cond_vols, 1e-15)
        std_resid = arr / safe_vols

        Q_bar = (std_resid.T @ std_resid) / T
        Q_bar = (Q_bar + Q_bar.T) / 2.0

        dcc_a, dcc_b = self._fit_dcc_params(std_resid, Q_bar)

        correlations = np.zeros((T, n, n), dtype=np.float64)
        covariances = np.zeros((T, n, n), dtype=np.float64)
        Q_t = Q_bar.copy()

        for t in range(T):
            if t > 0:
                eps_prev = std_resid[t - 1]
                Q_t = (
                    (1 - dcc_a - dcc_b) * Q_bar
                    + dcc_a * np.outer(eps_prev, eps_prev)
                    + dcc_b * Q_t
                )

            diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-15))
            R_t = Q_t / np.outer(diag_sqrt, diag_sqrt)
            np.fill_diagonal(R_t, 1.0)
            R_t = np.clip(R_t, -1.0, 1.0)
            R_t = (R_t + R_t.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(R_t)
            if eigvals[0] < 1e-8:
                eigvals = np.maximum(eigvals, 1e-8)
                R_t = eigvecs @ np.diag(eigvals) @ eigvecs.T
                np.fill_diagonal(R_t, 1.0)

            correlations[t] = R_t
            D_t = np.diag(cond_vols[t])
            covariances[t] = D_t @ R_t @ D_t

        self._result = DCCResult(
            covariances=covariances,
            correlations=correlations,
            conditional_vols=cond_vols,
            dcc_a=dcc_a,
            dcc_b=dcc_b,
            garch_params=garch_params,
        )
        self._fitted = True

        elapsed = (time.perf_counter() - t0) * 1000.0
        return SPTResult(
            data=self._result,
            metadata={
                "method": "DCC-GARCH(1,1)",
                "T": T,
                "n_assets": n,
                "dcc_a": dcc_a,
                "dcc_b": dcc_b,
                "persistence": dcc_a + dcc_b,
            },
            computation_time_ms=elapsed,
        )

    # -- CovarianceRateProcess interface ----------------------------------------

    def covariance_at(self, t: float) -> NDArray[np.float64]:
        """Return the covariance matrix at time t (nearest-neighbor lookup).

        Satisfies the ``quantspt.core.covariance.CovarianceRateProcess``
        protocol for seamless integration with γ* computation.

        Parameters
        ----------
        t : float
            Time point (matched to nearest observation).

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric PSD covariance matrix at time t.
        """
        require(self._fitted, "Must call .fit() before .covariance_at()")
        assert self._result is not None and self._times is not None
        idx = int(np.argmin(np.abs(self._times - t)))
        return self._result.covariances[idx].copy()

    def n_assets(self) -> int:
        """Number of assets in the fitted model."""
        require(self._fitted, "Must call .fit() before .n_assets()")
        assert self._result is not None
        return self._result.covariances.shape[1]

    # -- internal GARCH(1,1) ----------------------------------------------------

    def _fit_garch(
        self, returns: NDArray[np.float64]
    ) -> tuple[float, float, float, NDArray[np.float64]]:
        """Fit GARCH(1,1) to a single return series."""
        if self._use_arch:
            try:
                return self._fit_garch_arch(returns)
            except (ImportError, RuntimeError, ValueError):
                pass
        return self._fit_garch_internal(returns)

    def _fit_garch_arch(
        self, returns: NDArray[np.float64]
    ) -> tuple[float, float, float, NDArray[np.float64]]:
        """Fit GARCH(1,1) using the arch package."""
        from arch import arch_model

        am = arch_model(
            returns * 100, vol="Garch", p=1, q=1, mean="Zero", rescale=False
        )
        res = am.fit(disp="off", show_warning=False)
        omega = res.params["omega"] / 1e4
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]
        sigma2 = (res.conditional_volatility / 100.0) ** 2
        return omega, alpha, beta, sigma2

    def _fit_garch_internal(
        self, returns: NDArray[np.float64]
    ) -> tuple[float, float, float, NDArray[np.float64]]:
        """Simple GARCH(1,1) MLE fallback."""
        T = len(returns)
        var_r = np.var(returns)

        def neg_ll(params: NDArray[np.float64]) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.empty(T)
            sigma2[0] = var_r
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
            return -ll

        x0 = np.array([var_r * 0.05, 0.05, 0.90])
        bounds = [(1e-10, None), (1e-10, 0.5), (0.5, 0.9999)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = res.x

        sigma2 = np.empty(T)
        sigma2[0] = var_r
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        return omega, alpha, beta, sigma2

    # -- internal DCC estimation ------------------------------------------------

    def _fit_dcc_params(
        self,
        std_resid: NDArray[np.float64],
        Q_bar: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Estimate DCC parameters (a, b) via composite log-likelihood.

        Uses SLSQP with an explicit constraint a + b < 1 instead of
        returning a penalty wall (1e10) which can distort numerical
        gradients near the boundary.
        """
        T, _n = std_resid.shape

        def neg_ll(params: NDArray[np.float64]) -> float:
            a, b = params

            Q_t = Q_bar.copy()
            ll = 0.0
            for t in range(1, T):
                eps = std_resid[t - 1]
                Q_t = (1 - a - b) * Q_bar + a * np.outer(eps, eps) + b * Q_t

                diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-15))
                R_t = Q_t / np.outer(diag_sqrt, diag_sqrt)
                np.fill_diagonal(R_t, 1.0)
                R_t = (R_t + R_t.T) / 2.0

                try:
                    sign, logdet = np.linalg.slogdet(R_t)
                    if sign <= 0:
                        return 1e10
                    eps_t = std_resid[t]
                    R_inv = np.linalg.solve(R_t, eps_t)
                    ll -= 0.5 * (logdet + eps_t @ R_inv - eps_t @ eps_t)
                except np.linalg.LinAlgError:
                    return 1e10
            return -ll

        x0 = np.array([0.01, 0.95])
        bounds = [(1e-6, 0.3), (1e-6, 0.9999)]
        constraints = {"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1])}
        res = minimize(
            neg_ll, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        a, b = res.x
        if a + b >= 1:
            a, b = 0.01, 0.95
        return float(a), float(b)
