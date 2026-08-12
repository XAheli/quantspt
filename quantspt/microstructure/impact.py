"""Price impact models and optimal execution.

Implements Kyle's (1985) market microstructure model for estimating price
impact from signed order flow, and the Almgren-Chriss (2001) optimal
execution framework for computing mean-variance efficient liquidation
trajectories.

Mathematical References
-----------------------
- Kyle's lambda: Kyle (1985), "Continuous Auctions and Insider Trading,"
  Econometrica 53(6), pp. 1315-1335.
  Price change is linear in signed order flow:
    ΔP_t = λ Q_t + ε_t
  where Q_t is net signed volume and λ = (1/2) σ_v / σ_u reflects the
  market maker's adverse selection problem. Estimated via OLS.

- Almgren-Chriss optimal execution: Almgren & Chriss (2001), "Optimal
  Execution of Portfolio Transactions," Journal of Risk 3(2), pp. 5-39.
  Optimal trajectory minimizes E[cost] + lambda Var[cost]:
    x_k = X * sinh(kappa*(T - t_k)) / sinh(kappa*T)
  where kappa = sqrt(lambda_risk * sigma^2 / eta_tilde) balances risk
  aversion against temporary impact, X is total shares to liquidate,
  eta_tilde = eta - 0.5*gamma*tau (AC2001, Eq. 18) adjusts for
  discrete trading, and the trading rate is:
    n_k = (x_{k-1} - x_k) / tau
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult
from ..backtesting.execution import ExecutionResult

__all__ = [
    "AlmgrenChrissExecution",
    "AlmgrenChrissResult",
    "KyleLambdaResult",
    "estimate_kyle_lambda",
    "optimal_execution_trajectory",
]


@dataclass(frozen=True)
class KyleLambdaResult:
    """Result of Kyle's lambda estimation.

    Attributes
    ----------
    kyle_lambda : float
        Price impact coefficient λ (price change per unit of signed flow).
    intercept : float
        OLS intercept (should be near zero for efficient markets).
    r_squared : float
        R² of the regression.
    t_statistic : float
        t-statistic for λ.
    market_depth : float
        Inverse of λ, interpreted as market depth (1/λ).
    n_obs : int
        Number of observations used.
    """

    kyle_lambda: float
    intercept: float
    r_squared: float
    t_statistic: float
    market_depth: float
    n_obs: int


@dataclass(frozen=True)
class AlmgrenChrissResult:
    """Optimal execution trajectory from the Almgren-Chriss model.

    Attributes
    ----------
    trajectory : NDArray[np.float64]
        Optimal holdings x_k at each time step, shape ``(N+1,)``.
    trade_list : NDArray[np.float64]
        Shares to trade at each step n_k = x_{k-1} − x_k, shape ``(N,)``.
    times : NDArray[np.float64]
        Time grid, shape ``(N+1,)``.
    expected_cost : float
        E[total cost] of the optimal strategy.
    cost_variance : float
        Var[total cost] of the optimal strategy.
    kappa : float
        Urgency parameter kappa = sqrt(lambda * sigma^2 / eta_tilde),
        where eta_tilde = eta - 0.5*gamma*tau.
    """

    trajectory: NDArray[np.float64]
    trade_list: NDArray[np.float64]
    times: NDArray[np.float64]
    expected_cost: float
    cost_variance: float
    kappa: float


# ---------------------------------------------------------------------------
# Kyle's Lambda
# ---------------------------------------------------------------------------


def estimate_kyle_lambda(
    price_changes: NDArray[np.float64],
    signed_volume: NDArray[np.float64],
) -> SPTResult[KyleLambdaResult]:
    r"""Estimate Kyle's lambda via OLS regression.

    Fits the Kyle (1985) price impact model:

    .. math::
        \Delta P_t = \alpha + \lambda\,Q_t + \varepsilon_t

    where ΔP_t is the price change and Q_t is the signed order flow
    (positive for buys, negative for sells).

    Parameters
    ----------
    price_changes : ndarray of shape (T,)
        Price changes ΔP_t (or log-price changes).
    signed_volume : ndarray of shape (T,)
        Net signed volume Q_t. Positive values indicate net buying
        pressure; negative values indicate net selling pressure.

    Returns
    -------
    SPTResult[KyleLambdaResult]
        Estimated λ with regression diagnostics.

    References
    ----------
    Kyle (1985), "Continuous Auctions and Insider Trading,"
    Econometrica 53(6), pp. 1315-1335, Proposition 1.
    """
    t0 = time.perf_counter()
    price_changes = np.asarray(price_changes, dtype=np.float64).ravel()
    signed_volume = np.asarray(signed_volume, dtype=np.float64).ravel()
    T = len(price_changes)
    require(
        len(signed_volume) == T,
        "price_changes and signed_volume must have equal length",
    )
    require(T >= 10, f"Need at least 10 observations, got {T}")

    X = np.column_stack([np.ones(T), signed_volume])
    beta_hat = np.linalg.lstsq(X, price_changes, rcond=None)[0]
    intercept = float(beta_hat[0])
    kyle_lambda = float(beta_hat[1])

    residuals = price_changes - X @ beta_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((price_changes - price_changes.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse = ss_res / (T - 2)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_lambda = np.sqrt(mse * xtx_inv[1, 1])
    t_stat = kyle_lambda / se_lambda if se_lambda > 0 else 0.0

    depth = 1.0 / kyle_lambda if abs(kyle_lambda) > 1e-15 else float("inf")

    elapsed = (time.perf_counter() - t0) * 1000.0
    result = KyleLambdaResult(
        kyle_lambda=kyle_lambda,
        intercept=intercept,
        r_squared=r_squared,
        t_statistic=t_stat,
        market_depth=depth,
        n_obs=T,
    )
    return SPTResult(
        data=result,
        metadata={"method": "OLS", "T": T},
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Almgren-Chriss Optimal Execution
# ---------------------------------------------------------------------------


def optimal_execution_trajectory(
    total_shares: float,
    T: float,
    N: int,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
    risk_aversion: float = 1e-6,
) -> SPTResult[AlmgrenChrissResult]:
    r"""Compute the Almgren-Chriss optimal execution trajectory.

    Solves the mean-variance optimization (Almgren & Chriss, 2001, §3):

    .. math::
        \min_{x_0,\ldots,x_N}\; E[\text{cost}]
        + \lambda\,\text{Var}[\text{cost}]

    The optimal trajectory is (Eq. 20):

    .. math::
        x_k = X \cdot \frac{\sinh\bigl(\kappa\,(T - t_k)\bigr)}
        {\sinh(\kappa\,T)}

    where the urgency parameter is:

    .. math::
        \kappa = \sqrt{\frac{\lambda_{\text{risk}}\,\sigma^2}{\tilde{\eta}}}

    and the adjusted temporary impact (AC2001 Eq. 18) is:

    .. math::
        \tilde{\eta} = \eta - \tfrac{1}{2}\,\gamma\,\tau

    Parameters
    ----------
    total_shares : float
        X — total shares to liquidate (positive = selling).
    T : float
        Trading horizon (e.g., days).
    N : int
        Number of trading intervals.
    sigma : float
        Daily volatility of the asset.
    eta : float
        Temporary price impact parameter (cost per share per unit rate).
    gamma : float
        Permanent price impact parameter (cost per share squared).
    risk_aversion : float
        λ — risk aversion coefficient in the mean-variance objective.

    Returns
    -------
    SPTResult[AlmgrenChrissResult]
        Optimal trajectory, trade list, and cost statistics.

    References
    ----------
    Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions,"
    Journal of Risk 3(2), pp. 5-39, Eq. (18)-(23).
    """
    t0 = time.perf_counter()
    require(T > 0, f"T must be positive, got {T}")
    require(N >= 1, f"N must be >= 1, got {N}")
    require(sigma > 0, f"sigma must be positive, got {sigma}")
    require(eta > 0, f"eta must be positive, got {eta}")
    require(gamma >= 0, f"gamma must be non-negative, got {gamma}")
    require(
        risk_aversion >= 0, f"risk_aversion must be non-negative, got {risk_aversion}"
    )

    tau = T / N
    eta_tilde = eta - 0.5 * gamma * tau

    if risk_aversion > 0 and eta_tilde > 0:
        kappa_sq = risk_aversion * sigma**2 / eta_tilde
        kappa = np.sqrt(kappa_sq)
    else:
        kappa = 0.0

    times = np.linspace(0, T, N + 1).astype(np.float64)
    trajectory = np.zeros(N + 1, dtype=np.float64)

    if kappa > 1e-15 and kappa * T > 1e-15:
        sinh_kT = np.sinh(kappa * T)
        for k in range(N + 1):
            trajectory[k] = total_shares * np.sinh(kappa * (T - times[k])) / sinh_kT
    else:
        for k in range(N + 1):
            trajectory[k] = total_shares * (1.0 - times[k] / T)

    trajectory[0] = total_shares
    trajectory[-1] = 0.0

    trade_list = np.diff(-trajectory)

    expected_cost = 0.5 * gamma * total_shares**2
    for k in range(N):
        n_k = trade_list[k]
        expected_cost += eta_tilde / tau * n_k**2

    cost_variance = 0.0
    for k in range(N):
        cost_variance += sigma**2 * tau * trajectory[k + 1] ** 2

    elapsed = (time.perf_counter() - t0) * 1000.0
    result = AlmgrenChrissResult(
        trajectory=trajectory,
        trade_list=trade_list,
        times=times,
        expected_cost=expected_cost,
        cost_variance=cost_variance,
        kappa=kappa,
    )
    return SPTResult(
        data=result,
        metadata={
            "method": "Almgren-Chriss",
            "total_shares": total_shares,
            "T": T,
            "N": N,
            "sigma": sigma,
            "eta": eta,
            "gamma": gamma,
            "risk_aversion": risk_aversion,
        },
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# ExecutionModel integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlmgrenChrissExecution:
    r"""Almgren-Chriss execution model implementing ``ExecutionModel`` protocol.

    Computes execution cost by calling ``optimal_execution_trajectory()``
    for each asset's trade and accumulating the AC-computed expected cost.

    Parameters
    ----------
    eta : float
        Temporary impact parameter (per-share per-unit-rate).
    gamma : float
        Permanent impact parameter. Default 0 (temporary only).
    sigma : NDArray[np.float64]
        Per-asset daily volatility, shape ``(n,)``.
    risk_aversion : float
        Risk aversion for the optimal trajectory.
    trading_horizon : float
        Trading horizon in days (default 1 day).
    n_steps : int
        Number of trading intervals per horizon (default 5).

    References
    ----------
    Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions,"
    J. Risk 3(2), pp. 5-39.
    """

    eta: float
    gamma: float
    sigma: NDArray[np.float64]
    risk_aversion: float = 1e-6
    trading_horizon: float = 1.0
    n_steps: int = 5

    def __post_init__(self) -> None:
        require(self.eta > 0, f"eta must be positive, got {self.eta}")
        require(self.gamma >= 0, f"gamma must be non-negative, got {self.gamma}")

    def execute(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
    ) -> ExecutionResult:
        """Execute with Almgren-Chriss cost estimation.

        Computes the expected cost of optimally liquidating each
        asset's trade via ``optimal_execution_trajectory``.
        """
        require(portfolio_value > 0, "portfolio_value must be positive")
        delta_w = np.abs(target_weights - current_weights)
        total_cost = 0.0

        for i in range(len(delta_w)):
            if delta_w[i] < 1e-12:
                continue
            trade_shares = delta_w[i] * portfolio_value
            ac_result = optimal_execution_trajectory(
                total_shares=trade_shares,
                T=self.trading_horizon,
                N=self.n_steps,
                sigma=float(self.sigma[i]),
                eta=self.eta,
                gamma=self.gamma,
                risk_aversion=self.risk_aversion,
            )
            total_cost += ac_result.data.expected_cost / portfolio_value

        return ExecutionResult(weights=target_weights.copy(), cost=total_cost)
