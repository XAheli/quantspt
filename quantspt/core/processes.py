"""Stochastic process implementations and discretisation schemes.

The ``_typing`` module defines the :class:`StochasticProcess` and
:class:`Discretization` protocols.  This module provides concrete
implementations:

- **Euler-Maruyama** discretisation (strong order 0.5)
- **Milstein** discretisation (strong order 1.0 for 1-D)
- **Exact** discretisation for geometric Brownian motion
- **CorrelatedGBM** — a multivariate GBM implementing the protocol
- **simulate_path** — helper that wires a process + discretisation + RNG

Mathematical References
-----------------------
- GBM SDE: standard Itô calculus
- Euler-Maruyama: Kloeden & Platen, "Numerical Solution of SDEs" (1992)
- Milstein scheme: ibid., Theorem 10.3.5
- Exact GBM solution: dS = μS dt + σS dW ⟹ S(t) = S(0) exp((μ-σ²/2)t + σW(t))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "CorrelatedGBM",
    "EulerMaruyamaDiscretization",
    "ExactGBMDiscretization",
    "MilsteinDiscretization",
    "simulate_path",
]


# ---------------------------------------------------------------------------
# Discretisation schemes
# ---------------------------------------------------------------------------


class EulerMaruyamaDiscretization:
    r"""Euler-Maruyama discretisation for Itô SDEs.

    Given an SDE  dX = μ(t,X) dt + σ(t,X) dW, the Euler-Maruyama
    scheme advances the state by:

    .. math::
        X_{k+1} = X_k + \mu(t_k, X_k)\,\Delta t
                  + \sigma(t_k, X_k)\,\Delta W_k

    This has strong convergence order 0.5 and weak order 1.0.

    References
    ----------
    Kloeden & Platen (1992), §9.1
    """

    def evolve(
        self,
        process: object,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Advance one Euler-Maruyama step."""
        from .._typing import StochasticProcess

        assert isinstance(process, StochasticProcess)
        mu = process.drift(t0, x0)
        sigma = process.diffusion(t0, x0)
        return x0 + mu * dt + sigma @ dw


class MilsteinDiscretization:
    r"""Milstein discretisation for 1-D scalar SDEs.

    Adds the Lévy-area correction to Euler-Maruyama:

    .. math::
        X_{k+1} = X_k + \mu\,\Delta t + \sigma\,\Delta W
                  + \tfrac{1}{2}\,\sigma\,\sigma'\,
                    (\Delta W^2 - \Delta t)

    where σ' = ∂σ/∂x (approximated by finite differences when not
    supplied analytically).

    Strong convergence order 1.0 for scalar SDEs.

    Parameters
    ----------
    diffusion_deriv : callable, optional
        If provided, returns dσ/dx(t, x) as a scalar.  Otherwise
        a finite-difference approximation is used.
    fd_step : float
        Step size for finite-difference approximation of σ'.

    References
    ----------
    Kloeden & Platen (1992), Theorem 10.3.5
    """

    def __init__(
        self,
        diffusion_deriv: object | None = None,
        fd_step: float = 1e-6,
    ) -> None:
        self._diffusion_deriv = diffusion_deriv
        self._fd_step = fd_step

    def evolve(
        self,
        process: object,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Advance one Milstein step (1-D only)."""
        from .._typing import StochasticProcess

        assert isinstance(process, StochasticProcess)
        require(
            process.size() == 1,
            f"Milstein is implemented for 1-D processes, got size={process.size()}",
        )

        mu = process.drift(t0, x0)
        sigma = process.diffusion(t0, x0)  # shape (1, factors)
        sigma_val = sigma[0, 0]

        if self._diffusion_deriv is not None:
            from collections.abc import Callable

            assert isinstance(self._diffusion_deriv, Callable)  # type: ignore[arg-type]
            dsigma = float(self._diffusion_deriv(t0, x0))  # type: ignore[operator]
        else:
            h = self._fd_step
            x_plus = x0.copy()
            x_plus[0] += h
            x_minus = x0.copy()
            x_minus[0] -= h
            sig_plus = process.diffusion(t0, x_plus)[0, 0]
            sig_minus = process.diffusion(t0, x_minus)[0, 0]
            dsigma = (sig_plus - sig_minus) / (2.0 * h)

        euler = x0 + mu * dt + sigma @ dw
        milstein_correction = np.zeros_like(x0)
        milstein_correction[0] = 0.5 * sigma_val * dsigma * (dw[0] ** 2 - dt)
        return euler + milstein_correction


class ExactGBMDiscretization:
    r"""Exact discretisation for geometric Brownian motion.

    For a GBM  dS = μS dt + σS dW, the exact solution over [t, t+Δt] is:

    .. math::
        S(t + \Delta t) = S(t) \exp\bigl[
            (\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\,\Delta W
        \bigr]

    This works for multivariate GBM with diagonal or correlated noise
    when the process provides its own ``evolve`` method.  As a
    discretisation, it delegates to ``process.evolve``.

    References
    ----------
    Exact GBM solution from Itô's lemma applied to log S.
    """

    def evolve(
        self,
        process: object,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Delegate to the process's own exact evolve."""
        from .._typing import StochasticProcess

        assert isinstance(process, StochasticProcess)
        return process.evolve(t0, x0, dt, dw)


# ---------------------------------------------------------------------------
# Concrete process: Correlated GBM
# ---------------------------------------------------------------------------


@dataclass
class CorrelatedGBM:
    r"""Multivariate correlated geometric Brownian motion.

    Each component follows:

    .. math::
        dS_i = \mu_i S_i\,dt + S_i \sum_\nu L_{i\nu}\,dW_\nu

    where L is the Cholesky factor of the covariance matrix
    (so that a = L L^T).

    The exact solution for each path segment is:

    .. math::
        S_i(t+\Delta t) = S_i(t) \exp\bigl[
            (\mu_i - \tfrac{1}{2} a_{ii})\Delta t
            + \sum_\nu L_{i\nu}\,\Delta W_\nu
        \bigr]

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Drift vector (rates of return).
    cov : ndarray of shape (n, n)
        Covariance matrix (must be symmetric PSD).
    x0 : ndarray of shape (n,)
        Initial values S(0).  Must be positive.
    """

    mu: NDArray[np.float64]
    cov: NDArray[np.float64]
    x0: NDArray[np.float64]
    _cholesky: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.mu)
        require(self.cov.shape == (n, n), "Covariance shape mismatch")
        require(
            bool(np.all(self.x0 > 0)),
            f"Initial values must be positive, min={float(np.min(self.x0)):.2e}",
        )
        self._cholesky: NDArray[np.float64] = np.asarray(
            np.linalg.cholesky(self.cov), dtype=np.float64
        )

    def size(self) -> int:
        """Number of assets."""
        return len(self.mu)

    def factors(self) -> int:
        """Number of independent Brownian motions (= number of assets)."""
        return len(self.mu)

    def initial_values(self) -> NDArray[np.float64]:
        """Starting prices S(0)."""
        return self.x0.copy()

    def drift(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""GBM drift: μ_i · S_i."""
        return self.mu * x

    def diffusion(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""GBM diffusion: diag(S) · L, shape (n, n)."""
        return np.diag(x) @ self._cholesky

    def evolve(
        self,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Exact GBM step via log-normal transition."""
        a_diag = np.diag(self.cov)
        log_increment = (self.mu - 0.5 * a_diag) * dt + self._cholesky @ dw
        result: NDArray[np.float64] = x0 * np.exp(log_increment)
        return result


# ---------------------------------------------------------------------------
# Path simulation helper
# ---------------------------------------------------------------------------


def simulate_path(
    process: object,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    discretization: object | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Simulate a single path of a stochastic process.

    Parameters
    ----------
    process : StochasticProcess
        The SDE to simulate.  Must implement the
        :class:`~quantspt._typing.StochasticProcess` protocol.
    T : float
        Terminal time.
    n_steps : int
        Number of time steps.
    rng : numpy.random.Generator
        Random number generator.
    discretization : Discretization, optional
        Numerical scheme.  If ``None``, the process's own ``evolve``
        method is used (exact for GBM).

    Returns
    -------
    tuple of (times, path)
        ``times`` has shape (n_steps + 1,), ``path`` has shape
        (n_steps + 1, size).
    """
    from .._typing import StochasticProcess

    assert isinstance(process, StochasticProcess)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    n_factors = process.factors()
    n_size = process.size()

    times: NDArray[np.float64] = np.asarray(
        np.linspace(0.0, T, n_steps + 1), dtype=np.float64
    )
    path: NDArray[np.float64] = np.empty((n_steps + 1, n_size))
    path[0] = process.initial_values()

    for k in range(n_steps):
        dw = rng.standard_normal(n_factors) * sqrt_dt
        t_k = times[k]
        x_k = path[k]

        if discretization is not None:
            from .._typing import Discretization

            assert isinstance(discretization, Discretization)
            path[k + 1] = discretization.evolve(process, t_k, x_k, dt, dw)
        else:
            path[k + 1] = process.evolve(t_k, x_k, dt, dw)

    return times, path
