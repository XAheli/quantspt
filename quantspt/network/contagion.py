"""Shock propagation models for financial networks.

Implements two foundational contagion mechanisms: the Eisenberg-Noe (2001)
clearing payment vector for solvency cascades, and the linear DebtRank
algorithm (Battiston et al., 2012) for distress propagation through
exposure-weighted linkages.

Mathematical References
-----------------------
- Eisenberg-Noe clearing vector: Eisenberg & Noe (2001), "Systemic Risk
  in Financial Systems," Management Science 47(2), pp. 236-249.
  Fixed-point problem: p* = Π^T · min(e + Π · p*, d) where Π is the
  relative liability matrix, e is external assets, d is total obligations.
  Solved via Picard iteration (Theorem 2 of Eisenberg-Noe) which converges
  monotonically from above.

- Linear DebtRank: Battiston, Puliga, Kaushik, Tasca & Caldarelli (2012),
  "DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic
  Risk," Scientific Reports 2, Art. 541.
  Iterative distress propagation:
    h_i(t+1) = min{1, h_i(t) + Σ_j W_ij · h_j(t) · (1 − h_j(t−1))}
  where W_ij = exposure(i→j) / equity(i) is the relative exposure matrix,
  and h_i ∈ [0, 1] is the distress level of node i.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult

__all__ = [
    "ContagionResult",
    "DebtRank",
    "EisenbergNoe",
    "clearing_vector",
    "debt_rank",
]


@dataclass(frozen=True)
class ContagionResult:
    """Result of a contagion simulation.

    Attributes
    ----------
    distress : NDArray[np.float64]
        Final distress levels h_i ∈ [0, 1] for each node.
    defaults : NDArray[np.bool_]
        Boolean mask of defaulted nodes (distress = 1).
    n_defaults : int
        Number of defaulted nodes.
    total_loss : float
        Total system loss: Σ_i equity_i · h_i.
    iterations : int
        Number of iterations until convergence.
    distress_history : NDArray[np.float64]
        Distress levels at each iteration, shape ``(iterations+1, n)``.
    """

    distress: NDArray[np.float64]
    defaults: NDArray[np.bool_]
    n_defaults: int
    total_loss: float
    iterations: int
    distress_history: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Eisenberg-Noe Clearing Vector
# ---------------------------------------------------------------------------


class EisenbergNoe:
    r"""Eisenberg-Noe (2001) clearing payment vector model.

    Given a network of bilateral obligations, finds the unique clearing
    vector p* that satisfies:

    .. math::
        p_i^* = \min\bigl(d_i,\; e_i + \sum_j \Pi_{ji} \cdot p_j^*\bigr)

    where d_i is total obligations, e_i is external assets, and Π is the
    relative liability matrix (Π_{ij} = L_{ij} / d_i).

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        L_{ij} = liability of node i to node j.
    external_assets : ndarray of shape (n,)
        External (non-interbank) assets of each node.

    References
    ----------
    Eisenberg & Noe (2001), "Systemic Risk in Financial Systems,"
    Management Science 47(2), pp. 236-249, Theorem 2.
    """

    def __init__(
        self,
        liabilities: NDArray[np.float64],
        external_assets: NDArray[np.float64],
    ) -> None:
        liabilities = np.asarray(liabilities, dtype=np.float64)
        external_assets = np.asarray(external_assets, dtype=np.float64)
        n = liabilities.shape[0]
        require(liabilities.shape == (n, n), "liabilities must be square")
        require(external_assets.shape == (n,), f"external_assets shape must be ({n},)")
        require(
            bool(np.all(liabilities >= 0)),
            "liabilities must be non-negative",
        )
        require(
            bool(np.all(external_assets >= 0)),
            "external_assets must be non-negative",
        )

        self._liabilities = liabilities
        self._external_assets = external_assets
        self._n = n

        self._total_obligations = liabilities.sum(axis=1)

        safe = np.where(self._total_obligations > 0, self._total_obligations, 1.0)
        self._relative_liabilities = liabilities / safe[:, np.newaxis]

    def solve(
        self,
        *,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> SPTResult[ContagionResult]:
        """Find the clearing vector via Picard iteration.

        Starts from full payment (p = d) and iterates downward until
        convergence. Eisenberg-Noe Theorem 2 guarantees monotone
        convergence to the unique greatest clearing vector.

        Parameters
        ----------
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance on ‖p^(k+1) − p^(k)‖_∞.

        Returns
        -------
        SPTResult[ContagionResult]
            Clearing result with payment ratios converted to distress
            levels: h_i = 1 − p_i*/d_i.
        """
        t0 = time.perf_counter()
        d = self._total_obligations.copy()
        e = self._external_assets.copy()
        Pi = self._relative_liabilities
        n = self._n

        p = d.copy()
        history = [np.zeros(n, dtype=np.float64)]

        n_iters = 0
        diff = float("inf")
        for _ in range(max_iter):
            inflows = Pi.T @ p
            total_assets = e + inflows
            p_new = np.minimum(d, total_assets)
            p_new = np.maximum(p_new, 0.0)

            diff = float(np.max(np.abs(p_new - p)))
            safe_d = np.where(d > 0, d, 1.0)
            distress = 1.0 - p_new / safe_d
            distress = np.where(d > 0, distress, 0.0)
            history.append(distress.copy())
            n_iters += 1

            if diff < tol:
                p = p_new
                break
            p = p_new

        safe_d = np.where(d > 0, d, 1.0)
        distress = 1.0 - p / safe_d
        distress = np.where(d > 0, distress, 0.0)
        distress = np.clip(distress, 0.0, 1.0)
        defaults = distress > 1.0 - 1e-8

        equity = e + (Pi.T @ p) - d
        total_loss = float(np.sum(np.maximum(-equity, 0.0)))

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = ContagionResult(
            distress=distress,
            defaults=defaults,
            n_defaults=int(defaults.sum()),
            total_loss=total_loss,
            iterations=n_iters,
            distress_history=np.array(history),
        )
        return SPTResult(
            data=result,
            metadata={
                "method": "Eisenberg-Noe",
                "converged": diff < tol,
                "final_diff": diff,
            },
            computation_time_ms=elapsed,
        )


def clearing_vector(
    liabilities: NDArray[np.float64],
    external_assets: NDArray[np.float64],
    **kwargs: Any,
) -> SPTResult[ContagionResult]:
    """Convenience wrapper for :class:`EisenbergNoe`.solve().

    Parameters
    ----------
    liabilities : ndarray of shape (n, n)
        Bilateral liability matrix.
    external_assets : ndarray of shape (n,)
        External assets per node.
    **kwargs
        Forwarded to ``EisenbergNoe.solve()``.

    Returns
    -------
    SPTResult[ContagionResult]
    """
    return EisenbergNoe(liabilities, external_assets).solve(**kwargs)


# ---------------------------------------------------------------------------
# DebtRank
# ---------------------------------------------------------------------------


class DebtRank:
    r"""Linear DebtRank algorithm (Battiston et al., 2012).

    Iterative distress propagation on a network of financial exposures:

    .. math::
        h_i(t+1) = \min\Bigl\{1,\; h_i(t) + \sum_j W_{ij} \cdot h_j(t)
                     \cdot \bigl(1 - h_j(t-1)\bigr)\Bigr\}

    where W_{ij} = exposure(i→j) / equity(i) is the impact weight, and
    h_i ∈ [0, 1] measures distress from healthy (0) to default (1).

    The term (1 − h_j(t−1)) prevents double-counting: a node only
    transmits its *incremental* distress at each step.

    Parameters
    ----------
    exposure_matrix : ndarray of shape (n, n)
        W_{ij} = exposure of node i to node j. Does NOT need to be
        normalized by equity — the constructor handles that.
    equity : ndarray of shape (n,)
        Equity capital of each node.

    References
    ----------
    Battiston, Puliga, Kaushik, Tasca & Caldarelli (2012),
    "DebtRank: Too Central to Fail?," Scientific Reports 2, Art. 541, Eq. (3).
    """

    def __init__(
        self,
        exposure_matrix: NDArray[np.float64],
        equity: NDArray[np.float64],
    ) -> None:
        exposure_matrix = np.asarray(exposure_matrix, dtype=np.float64)
        equity = np.asarray(equity, dtype=np.float64)
        n = exposure_matrix.shape[0]
        require(exposure_matrix.shape == (n, n), "exposure_matrix must be square")
        require(equity.shape == (n,), f"equity shape must be ({n},)")
        require(bool(np.all(equity > 0)), "equity must be strictly positive")

        safe_eq = np.where(equity > 0, equity, 1.0)
        self._W = exposure_matrix / safe_eq[:, np.newaxis]
        self._equity = equity
        self._n = n

    def propagate(
        self,
        initial_shock: NDArray[np.float64],
        *,
        max_rounds: int = 100,
    ) -> SPTResult[ContagionResult]:
        r"""Propagate an initial distress shock through the network.

        Parameters
        ----------
        initial_shock : ndarray of shape (n,)
            Initial distress h_i(0) ∈ [0, 1]. Shocked nodes should have
            h_i(0) > 0; unshocked nodes should be 0.
        max_rounds : int
            Maximum propagation rounds.

        Returns
        -------
        SPTResult[ContagionResult]
            Converged distress levels and default indicators.
        """
        t0 = time.perf_counter()
        initial_shock = np.asarray(initial_shock, dtype=np.float64)
        require(
            initial_shock.shape == (self._n,),
            f"initial_shock shape must be ({self._n},)",
        )
        require(
            bool(np.all((initial_shock >= 0) & (initial_shock <= 1))),
            "initial_shock values must be in [0, 1]",
        )

        h_prev: NDArray[np.float64] = np.zeros(self._n, dtype=np.float64)
        h_curr: NDArray[np.float64] = np.clip(initial_shock.copy(), 0.0, 1.0)
        history = [h_curr.copy()]

        n_rounds = 0
        for _ in range(max_rounds):
            incremental = h_curr * (1.0 - h_prev)
            contagion = self._W @ incremental
            h_next: NDArray[np.float64] = np.minimum(1.0, h_curr + contagion)
            n_rounds += 1

            if np.max(np.abs(h_next - h_curr)) < 1e-12:
                h_prev = h_curr
                h_curr = h_next
                history.append(h_curr.copy())
                break

            h_prev = h_curr
            h_curr = h_next
            history.append(h_curr.copy())

        defaults = h_curr > 1.0 - 1e-8
        total_loss = float(np.sum(self._equity * h_curr))

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = ContagionResult(
            distress=h_curr,
            defaults=defaults,
            n_defaults=int(defaults.sum()),
            total_loss=total_loss,
            iterations=n_rounds,
            distress_history=np.array(history),
        )
        return SPTResult(
            data=result,
            metadata={"method": "LinearDebtRank", "rounds": n_rounds},
            computation_time_ms=elapsed,
        )


def debt_rank(
    exposure_matrix: NDArray[np.float64],
    equity: NDArray[np.float64],
    initial_shock: NDArray[np.float64],
    **kwargs: Any,
) -> SPTResult[ContagionResult]:
    """Convenience wrapper for :class:`DebtRank`.propagate().

    Parameters
    ----------
    exposure_matrix : ndarray of shape (n, n)
        Bilateral exposure matrix.
    equity : ndarray of shape (n,)
        Equity per node.
    initial_shock : ndarray of shape (n,)
        Initial distress levels.
    **kwargs
        Forwarded to ``DebtRank.propagate()``.

    Returns
    -------
    SPTResult[ContagionResult]

    References
    ----------
    Battiston et al. (2012), "DebtRank: Too Central to Fail?,"
    Scientific Reports 2, Art. 541.
    """
    return DebtRank(exposure_matrix, equity).propagate(initial_shock, **kwargs)
