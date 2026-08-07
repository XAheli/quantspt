"""Covariance rate processes and relative covariance.

This module implements the instantaneous covariance rate a_{ij}(t) and the
relative covariance matrix τ^π_{ij}(t), which are the fundamental objects
that drive excess growth and portfolio performance in SPT.

Mathematical References
-----------------------
- Covariance rate: F&K Survey Eq. 1.3, FKK Eq. 2.5
- Relative covariance: F&K Survey Eq. 1.19, FKK Eq. 5.3
- Non-degeneracy condition: FKK Eq. 2.3
- Bounds on τ^π: FKK Eq. 5.10-5.12
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "CovarianceRateProcess",
    "non_degeneracy_bounds",
    "portfolio_covariance_vector",
    "portfolio_variance",
    "relative_covariance",
    "tau_bounds",
    "tau_diagonal",
    "verify_non_degeneracy",
]


@runtime_checkable
class CovarianceRateProcess(Protocol):
    r"""Protocol for time-varying covariance rate processes.

    Defines the interface for objects that provide an instantaneous
    covariance rate matrix a(t) at any point in time. Implementations
    may be based on constant matrices, rolling estimates, model-implied
    rates, or stochastic volatility processes.

    The covariance rate a_{ij}(t) satisfies:
    d⟨log X_i, log X_j⟩(t) = a_{ij}(t) dt

    Implementers must return a symmetric positive semi-definite matrix.
    """

    def covariance_at(self, t: float) -> NDArray[np.float64]:
        """Return the covariance rate matrix at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric PSD covariance rate matrix.
        """
        ...

    def n_assets(self) -> int:
        """Number of assets in the covariance matrix."""
        ...


class ConstantCovarianceRate:
    r"""Constant (time-homogeneous) covariance rate process.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Fixed covariance rate matrix.
    """

    def __init__(self, a: NDArray[np.float64]) -> None:
        require(a.ndim == 2, f"Covariance must be 2-D, got shape {a.shape}")
        require(a.shape[0] == a.shape[1], "Covariance must be square")
        self._a = a

    def covariance_at(self, t: float) -> NDArray[np.float64]:
        """Return constant covariance matrix regardless of t."""
        return self._a.copy()

    def n_assets(self) -> int:
        """Number of assets."""
        return self._a.shape[0]


class RollingCovarianceRate:
    r"""Time-varying covariance rate from rolling window estimation.

    Stores pre-computed covariance matrices at discrete time points
    and interpolates for intermediate queries.

    Parameters
    ----------
    times : ndarray of shape (T,)
        Time points at which covariance was estimated.
    covariances : ndarray of shape (T, n, n)
        Covariance rate matrices at each time point.
    interpolation : str
        ``'nearest'`` (default) or ``'linear'``.
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        covariances: NDArray[np.float64],
        interpolation: str = "nearest",
    ) -> None:
        require(times.ndim == 1, f"times must be 1-D, got shape {times.shape}")
        require(
            covariances.ndim == 3,
            f"covariances must be 3-D, got shape {covariances.shape}",
        )
        require(
            len(times) == covariances.shape[0],
            f"Length mismatch: {len(times)} times, {covariances.shape[0]} matrices",
        )
        require(
            interpolation in ("nearest", "linear"),
            f"interpolation must be 'nearest' or 'linear', got '{interpolation}'",
        )
        self._times = times
        self._covariances = covariances
        self._interpolation = interpolation

    def covariance_at(self, t: float) -> NDArray[np.float64]:
        """Retrieve or interpolate covariance at time t."""
        if self._interpolation == "nearest":
            idx = int(np.argmin(np.abs(self._times - t)))
            return self._covariances[idx].copy()
        else:
            idx = int(np.searchsorted(self._times, t))
            if idx == 0:
                return self._covariances[0].copy()
            if idx >= len(self._times):
                return self._covariances[-1].copy()
            t0, t1 = self._times[idx - 1], self._times[idx]
            alpha = (t - t0) / (t1 - t0)
            result = (1.0 - alpha) * self._covariances[
                idx - 1
            ] + alpha * self._covariances[idx]
            return result

    def n_assets(self) -> int:
        """Number of assets."""
        return self._covariances.shape[1]


def relative_covariance(
    a: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the relative covariance matrix τ^π.

    .. math::
        \tau^{\pi}_{ij}(t) = a_{ij} - a^{\pi}_i - a^{\pi}_j + a_{\pi\pi}

    Equivalently:

    .. math::
        \tau^{\pi}_{ij} = (\pi - e_i)^T a (\pi - e_j)

    This matrix measures the covariance of stock returns *relative to* the
    portfolio π. It is positive semi-definite with π in its null space.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix. Must be symmetric PSD.
    pi : ndarray of shape (n,)
        Portfolio weights, must sum to 1.

    Returns
    -------
    ndarray of shape (n, n)
        Relative covariance matrix τ^π.

    Notes
    -----
    Key properties (F&K Lemma 3.1):
      - τ^π is positive semi-definite
      - τ^π · π = 0 (π is in the null space)
      - For the market portfolio μ, τ^μ_{ii} = variance of stock i relative
        to the market

    Complexity: O(n²) via single matrix-vector product + broadcasts.

    References
    ----------
    F&K Survey Eq. 1.19, FKK Eq. 5.3

    Examples
    --------
    >>> a = np.array([[0.04, 0.01], [0.01, 0.04]])
    >>> pi = np.array([0.6, 0.4])
    >>> tau = relative_covariance(a, pi)
    >>> np.allclose(tau @ pi, 0, atol=1e-14)
    True
    """
    n = len(pi)
    require(
        a.shape == (n, n),
        f"Covariance matrix shape {a.shape} incompatible with {n} assets",
    )
    require(
        bool(abs(float(pi.sum()) - 1.0) < 1e-8),
        f"Weights must sum to 1, got {pi.sum():.8f}",
    )

    a_pi = a @ pi  # (n,) vector: a^π_i = Σ_j π_j a_{ij}
    a_pipi = float(pi @ a_pi)  # scalar: π'aπ

    tau: NDArray[np.float64] = a - np.add.outer(a_pi, a_pi) + a_pipi
    return tau


def portfolio_variance(
    a: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> float:
    r"""Compute portfolio variance a_{ππ}(t) = π' a(t) π.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.
    pi : ndarray of shape (n,)
        Portfolio weights.

    Returns
    -------
    float
        Non-negative portfolio variance rate.

    References
    ----------
    F&K Survey Eq. 1.20
    """
    return float(pi @ a @ pi)


def portfolio_covariance_vector(
    a: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute a^π_i = Σ_j π_j a_{ij} for all i.

    This is the covariance of each stock with the portfolio π.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.
    pi : ndarray of shape (n,)
        Portfolio weights.

    Returns
    -------
    ndarray of shape (n,)
        Covariance of each stock with portfolio π.

    References
    ----------
    F&K Survey Eq. 1.20
    """
    return a @ pi


def non_degeneracy_bounds(
    a: NDArray[np.float64],
) -> tuple[float, float]:
    r"""Compute non-degeneracy bounds (ε, M) for covariance matrix.

    Find ε and M such that:

    .. math::
        \varepsilon \|\xi\|^2 \leq \xi' a \xi \leq M \|\xi\|^2
        \quad \forall \xi \in \mathbb{R}^n

    These are the smallest and largest eigenvalues of a.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Symmetric covariance rate matrix.

    Returns
    -------
    tuple of (float, float)
        (ε, M) — minimum and maximum eigenvalues.

    References
    ----------
    FKK Eq. 2.3
    """
    eigenvalues = np.linalg.eigvalsh(a)
    eps = float(eigenvalues[0])
    M = float(eigenvalues[-1])
    return eps, M


def verify_non_degeneracy(
    a: NDArray[np.float64],
    tol: float = 1e-10,
) -> bool:
    r"""Check whether a satisfies the non-degeneracy condition (FKK Eq. 2.3).

    A market model is non-degenerate if there exists ε > 0 such that
    ε‖ξ‖² ≤ ξ'a(t)ξ for all ξ ∈ ℝⁿ, i.e., the covariance matrix is
    strictly positive definite.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Covariance rate matrix.
    tol : float
        Minimum acceptable smallest eigenvalue.

    Returns
    -------
    bool
        True if a is positive definite with smallest eigenvalue > tol.

    References
    ----------
    FKK Eq. 2.3
    """
    eps, _ = non_degeneracy_bounds(a)
    return eps > tol


def tau_diagonal(
    a: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute diagonal of τ^π efficiently without forming full matrix.

    .. math::
        \tau^{\pi}_{ii} = a_{ii} - 2 a^{\pi}_i + a_{\pi\pi}

    This is sufficient for the excess growth rate computation.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Covariance rate matrix.
    pi : ndarray of shape (n,)
        Portfolio weights.

    Returns
    -------
    ndarray of shape (n,)
        Diagonal entries [τ^π_{11}, ..., τ^π_{nn}].

    References
    ----------
    Derived from F&K Survey Eq. 1.19
    """
    a_pi = a @ pi
    a_pipi = float(pi @ a_pi)
    return np.diag(a) - 2.0 * a_pi + a_pipi


def tau_bounds(
    pi: NDArray[np.float64],
    eps: float,
    M: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute theoretical bounds on τ^π_{ii} under non-degeneracy.

    Under the non-degeneracy condition (FKK Eq. 2.3) with constants ε, M:

    .. math::
        \varepsilon (1 - \pi_i)^2 \leq \tau^{\pi}_{ii}
        \leq M (1 - \pi_i)(2 - \pi_i)

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    eps : float
        Lower non-degeneracy constant (smallest eigenvalue of a).
    M : float
        Upper non-degeneracy constant (largest eigenvalue of a).

    Returns
    -------
    tuple of (ndarray, ndarray)
        (lower_bounds, upper_bounds) each of shape (n,).

    References
    ----------
    FKK Eq. 5.10
    """
    lower = eps * (1.0 - pi) ** 2
    upper = M * (1.0 - pi) * (2.0 - pi)
    return lower, upper
