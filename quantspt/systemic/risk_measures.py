"""Systemic risk measures: CoVaR, MES, and SRISK.

Quantifies how individual asset distress contributes to system-wide risk.
CoVaR measures the tail dependence between an institution and the system;
MES captures the expected loss of an asset when the system is in distress;
SRISK combines MES with leverage to estimate capital shortfall.

Mathematical References
-----------------------
- CoVaR: Adrian & Brunnermeier (2016), "CoVaR," American Economic Review
  106(7), pp. 1705-1741.
  CoVaR^{sys|i}_q is the q-quantile of the system return conditional on
  institution i being at its VaR:
    R_{sys,t} = α_q + β_q R_{i,t} + γ'_q M_t + ε_t   (quantile regression at q)
  ΔCoVaR = CoVaR^{sys|i=VaR_q} − CoVaR^{sys|i=median}

- MES: Acharya, Pedersen, Philippon & Richardson (2017), "Measuring
  Systemic Risk," Review of Financial Studies 30(1), pp. 2-47.
  MES_i = E[R_i | R_sys < VaR_q(R_sys)]
  Short-run MES estimates the expected loss of asset i when the market
  is in its left tail.

- SRISK: Brownlees & Engle (2017), "SRISK: A Conditional Capital
  Shortfall Measure of Systemic Risk," Review of Financial Studies
  30(1), pp. 48-79.
  SRISK_i = max{0, k(D_i + W_i) − W_i(1 − LRMES_i)}
  where LRMES = 1 − exp(−18 × MES), k is the prudential capital ratio,
  D is book debt, W is market equity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult

__all__ = [
    "CoVaRResult",
    "MESResult",
    "compute_covar",
    "compute_delta_covar",
    "compute_mes",
    "compute_srisk",
]


@dataclass(frozen=True)
class CoVaRResult:
    """CoVaR estimation result.

    Attributes
    ----------
    covar : float
        CoVaR^{sys|i}_q — system VaR conditional on institution distress.
    var_i : float
        VaR_q of institution i.
    alpha : float
        Quantile regression intercept.
    beta : float
        Quantile regression slope (sensitivity of system to institution).
    gamma : NDArray[np.float64] | None
        Coefficients on conditioning variables M_t, if provided.
    quantile : float
        Quantile level q used.
    """

    covar: float
    var_i: float
    alpha: float
    beta: float
    gamma: NDArray[np.float64] | None
    quantile: float


@dataclass(frozen=True)
class MESResult:
    """Marginal Expected Shortfall result.

    Attributes
    ----------
    mes : float
        MES_i = -E[R_i | R_sys < VaR_q(R_sys)] (positive = loss).
    var_sys : float
        System VaR at quantile q.
    n_tail_obs : int
        Number of observations in the tail.
    component_mes : NDArray[np.float64] | None
        MES for each asset (when computed for a panel).
    """

    mes: float
    var_sys: float
    n_tail_obs: int
    component_mes: NDArray[np.float64] | None = None


# ---------------------------------------------------------------------------
# CoVaR
# ---------------------------------------------------------------------------


def compute_covar(
    system_returns: NDArray[np.float64],
    institution_returns: NDArray[np.float64],
    *,
    quantile: float = 0.05,
    conditioning_vars: NDArray[np.float64] | None = None,
) -> SPTResult[CoVaRResult]:
    r"""Estimate CoVaR via quantile regression.

    Estimates the quantile regression (Adrian & Brunnermeier, 2016, Eq. 3):

    .. math::
        R_{\text{sys},t} = \alpha_q + \beta_q\,R_{i,t}
        + \gamma_q'\,M_t + \varepsilon_t

    at quantile q. Then:

    .. math::
        \text{CoVaR}^{\text{sys}|i}_q = \hat{\alpha}_q
        + \hat{\beta}_q\,\text{VaR}_q(R_i)
        + \hat{\gamma}_q'\,\bar{M}

    Parameters
    ----------
    system_returns : ndarray of shape (T,)
        System (market index) return series.
    institution_returns : ndarray of shape (T,)
        Individual institution return series.
    quantile : float
        Quantile level q (default 0.05 for 5% left tail).
    conditioning_vars : ndarray of shape (T, k), optional
        State variables M_t (e.g., VIX, credit spread, yield slope).

    Returns
    -------
    SPTResult[CoVaRResult]
        Fitted CoVaR with regression coefficients.

    References
    ----------
    Adrian & Brunnermeier (2016), "CoVaR," AER 106(7), Eq. (3)-(6).
    """
    t0 = time.perf_counter()
    system_returns = np.asarray(system_returns, dtype=np.float64).ravel()
    institution_returns = np.asarray(institution_returns, dtype=np.float64).ravel()
    T = len(system_returns)
    require(len(institution_returns) == T, "Return series must have equal length")
    require(T >= 30, f"Need at least 30 observations, got {T}")
    require(0 < quantile < 1, f"quantile must be in (0, 1), got {quantile}")

    import statsmodels.api as sm

    if conditioning_vars is not None:
        conditioning_vars = np.asarray(conditioning_vars, dtype=np.float64)
        require(
            conditioning_vars.shape[0] == T,
            f"conditioning_vars rows ({conditioning_vars.shape[0]}) != T ({T})",
        )
        X = np.column_stack([institution_returns, conditioning_vars])
    else:
        X = institution_returns.reshape(-1, 1)

    X = sm.add_constant(X)

    model = sm.QuantReg(system_returns, X)
    result = model.fit(q=quantile, max_iter=1000)

    alpha_hat = float(result.params[0])
    beta_hat = float(result.params[1])
    gamma_hat = result.params[2:] if len(result.params) > 2 else None

    var_i = float(np.quantile(institution_returns, quantile))

    covar_value = alpha_hat + beta_hat * var_i
    if gamma_hat is not None and conditioning_vars is not None:
        covar_value += float(gamma_hat @ conditioning_vars.mean(axis=0))

    elapsed = (time.perf_counter() - t0) * 1000.0
    covar_result = CoVaRResult(
        covar=covar_value,
        var_i=var_i,
        alpha=alpha_hat,
        beta=beta_hat,
        gamma=gamma_hat,
        quantile=quantile,
    )
    return SPTResult(
        data=covar_result,
        metadata={"method": "QuantileRegression", "quantile": quantile, "T": T},
        computation_time_ms=elapsed,
    )


def compute_delta_covar(
    system_returns: NDArray[np.float64],
    institution_returns: NDArray[np.float64],
    *,
    quantile: float = 0.05,
    conditioning_vars: NDArray[np.float64] | None = None,
) -> SPTResult[float]:
    r"""Compute ΔCoVaR — the marginal systemic risk contribution.

    .. math::
        \Delta\text{CoVaR}^{i}_q = \text{CoVaR}^{\text{sys}|i=\text{VaR}_q}
        - \text{CoVaR}^{\text{sys}|i=\text{median}}
        = \hat{\beta}_q \cdot \bigl(\text{VaR}_q(R_i) - \text{Median}(R_i)\bigr)

    Parameters
    ----------
    system_returns : ndarray of shape (T,)
        System return series.
    institution_returns : ndarray of shape (T,)
        Institution return series.
    quantile : float
        Quantile level.
    conditioning_vars : ndarray of shape (T, k), optional
        State variables.

    Returns
    -------
    SPTResult[float]
        ΔCoVaR value (negative means institution adds systemic risk).

    References
    ----------
    Adrian & Brunnermeier (2016), "CoVaR," AER 106(7), Eq. (6).
    """
    t0 = time.perf_counter()
    covar_result = compute_covar(
        system_returns,
        institution_returns,
        quantile=quantile,
        conditioning_vars=conditioning_vars,
    )
    cr = covar_result.data

    median_i = float(np.median(institution_returns))
    covar_at_median = cr.alpha + cr.beta * median_i
    if cr.gamma is not None and conditioning_vars is not None:
        covar_at_median += float(cr.gamma @ np.asarray(conditioning_vars).mean(axis=0))

    delta = cr.covar - covar_at_median

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SPTResult(
        data=delta,
        metadata={
            "method": "DeltaCoVaR",
            "covar_distress": cr.covar,
            "covar_median": covar_at_median,
            "beta": cr.beta,
        },
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# MES
# ---------------------------------------------------------------------------


def compute_mes(
    system_returns: NDArray[np.float64],
    asset_returns: NDArray[np.float64],
    *,
    quantile: float = 0.05,
) -> SPTResult[MESResult]:
    r"""Compute Marginal Expected Shortfall (as a positive loss).

    .. math::
        \text{MES}_i = -\,\mathbb{E}\bigl[R_i \mid R_{\text{sys}}
        < \text{VaR}_q(R_{\text{sys}})\bigr]

    Returns a **positive** value representing the expected loss (not
    the expected return) during system tail events, following the
    convention in Acharya et al. (2017) and Brownlees & Engle (2017).
    This ensures consistent sign when MES feeds into the LRMES/SRISK
    calculation.

    Parameters
    ----------
    system_returns : ndarray of shape (T,)
        System (market index) return series.
    asset_returns : ndarray of shape (T,) or (T, n)
        Individual asset returns. If 2-D, computes MES for each column.
    quantile : float
        Quantile level (default 0.05 for 5% worst days).

    Returns
    -------
    SPTResult[MESResult]
        MES estimate with tail statistics.

    References
    ----------
    Acharya, Pedersen, Philippon & Richardson (2017), "Measuring Systemic
    Risk," RFS 30(1), pp. 2-47, Definition 1.
    """
    t0 = time.perf_counter()
    system_returns = np.asarray(system_returns, dtype=np.float64).ravel()
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    T = len(system_returns)
    require(0 < quantile < 1, f"quantile must be in (0, 1), got {quantile}")

    if asset_returns.ndim == 1:
        require(
            len(asset_returns) == T, "asset_returns length must match system_returns"
        )
        single = True
        asset_returns = asset_returns.reshape(-1, 1)
    else:
        require(
            asset_returns.shape[0] == T, "First dimension must match system_returns"
        )
        single = False

    var_sys = float(np.quantile(system_returns, quantile))
    tail_mask = system_returns <= var_sys
    n_tail = int(tail_mask.sum())

    if n_tail == 0:
        mes_values = np.zeros(asset_returns.shape[1])
    else:
        mes_values = -asset_returns[tail_mask].mean(axis=0)

    elapsed = (time.perf_counter() - t0) * 1000.0

    mes_scalar = float(mes_values[0]) if single else float(mes_values.mean())
    result = MESResult(
        mes=mes_scalar,
        var_sys=var_sys,
        n_tail_obs=n_tail,
        component_mes=mes_values if not single else None,
    )
    return SPTResult(
        data=result,
        metadata={
            "method": "NonparametricMES",
            "quantile": quantile,
            "T": T,
            "n_tail": n_tail,
        },
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# SRISK
# ---------------------------------------------------------------------------


def compute_srisk(
    mes: float,
    book_debt: float,
    market_equity: float,
    *,
    capital_ratio: float = 0.08,
    horizon_days: int = 22 * 6,
) -> SPTResult[float]:
    r"""Compute SRISK — Systemic Risk Index.

    Uses the Brownlees & Engle (2017) formulation:

    .. math::
        \text{LRMES}_i = 1 - \exp(-18 \times \text{MES}_i)

    .. math::
        \widetilde{W}_i = W_i \,(1 - \text{LRMES}_i)

    .. math::
        \text{SRISK}_i = \max\bigl\{0,\;
        k\,(D_i + \widetilde{W}_i) - \widetilde{W}_i\bigr\}

    where MES is a positive loss, the factor 18 is the empirical
    scaling from daily MES to 6-month expected equity loss, and
    W-tilde is the distressed (post-crisis) equity value.

    Parameters
    ----------
    mes : float
        Short-run MES (positive loss convention, as returned by
        ``compute_mes``).
    book_debt : float
        Book value of debt D.
    market_equity : float
        Market capitalization W.
    capital_ratio : float
        Prudential capital ratio k (Basel III: 0.08).
    horizon_days : int
        Not used (LRMES uses the empirical factor of 18). Kept for
        API compatibility.

    Returns
    -------
    SPTResult[float]
        SRISK value (positive = capital shortfall).

    References
    ----------
    Brownlees & Engle (2017), "SRISK: A Conditional Capital Shortfall
    Measure of Systemic Risk," RFS 30(1), pp. 48-79, Eq. (4)-(6).
    """
    t0 = time.perf_counter()
    require(market_equity > 0, f"market_equity must be positive, got {market_equity}")
    require(book_debt >= 0, f"book_debt must be non-negative, got {book_debt}")

    lrmes = 1.0 - np.exp(-18.0 * mes)
    lrmes = float(np.clip(lrmes, 0.0, 1.0))

    distressed_equity = market_equity * (1.0 - lrmes)
    srisk = max(
        0.0, capital_ratio * (book_debt + distressed_equity) - distressed_equity
    )

    elapsed = (time.perf_counter() - t0) * 1000.0
    return SPTResult(
        data=srisk,
        metadata={
            "method": "SRISK",
            "lrmes": float(lrmes),
            "capital_ratio": capital_ratio,
            "horizon_days": horizon_days,
            "leverage": (book_debt + market_equity) / market_equity,
        },
        computation_time_ms=elapsed,
    )
