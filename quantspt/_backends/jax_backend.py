"""JAX compute backend — GPU acceleration and automatic differentiation.

Provides JIT-compiled implementations of core SPT operations using JAX.
Enables automatic gradient/Hessian computation for generating functions
and GPU acceleration for large-scale portfolio computations.

Requires the ``gpu`` extra: ``pip install quantspt[gpu]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["JaxBackend"]


def _require_jax() -> Any:
    """Import JAX or raise with installation instructions."""
    try:
        import jax
        import jax.numpy as jnp

        return jax, jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for the JAX backend. "
            "Install with: pip install quantspt[gpu]"
        ) from exc


class JaxBackend:
    """JAX-based implementation with JIT compilation and autodiff.

    All operations are JIT-compiled on first call. Results are returned
    as NumPy arrays for compatibility with the rest of the library.
    """

    def __init__(self) -> None:
        jax, jnp = _require_jax()
        self._jax = jax
        self._jnp = jnp
        self._jit_fns: dict[str, Any] = {}
        self._build_jit_functions()

    def _build_jit_functions(self) -> None:
        """Pre-compile core operations with JAX JIT."""
        jax = self._jax
        jnp = self._jnp

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _excess_growth_rate(pi: Any, a: Any) -> Any:  # pragma: no cover
            weighted_var = jnp.dot(pi, jnp.diag(a))
            port_var = pi @ a @ pi
            return 0.5 * (weighted_var - port_var)

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _relative_covariance(a: Any, pi: Any) -> Any:  # pragma: no cover
            a_pi = a @ pi
            a_pipi = pi @ a_pi
            return a - (a_pi[:, None] + a_pi[None, :]) + a_pipi

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _portfolio_variance(a: Any, pi: Any) -> Any:  # pragma: no cover
            return pi @ a @ pi

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _simulate_gbm_step(
            x: Any, mu: Any, cholesky: Any, dt: Any, dw: Any
        ) -> Any:  # pragma: no cover
            a_diag = jnp.sum(cholesky**2, axis=1)
            log_inc = (mu - 0.5 * a_diag) * dt + cholesky @ dw
            return x * jnp.exp(log_inc)

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _diversity_weights(mu: Any, p: Any) -> Any:  # pragma: no cover
            mu_p = mu**p
            return mu_p / jnp.sum(mu_p)

        self._jit_fns["excess_growth_rate"] = _excess_growth_rate
        self._jit_fns["relative_covariance"] = _relative_covariance
        self._jit_fns["portfolio_variance"] = _portfolio_variance
        self._jit_fns["simulate_gbm_step"] = _simulate_gbm_step
        self._jit_fns["diversity_weights"] = _diversity_weights

    @property
    def name(self) -> str:
        return "jax"

    def excess_growth_rate(
        self,
        pi: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> float:
        r"""JIT-compiled excess growth rate γ*_π."""
        jnp = self._jnp
        result = self._jit_fns["excess_growth_rate"](jnp.array(pi), jnp.array(a))
        return float(result)

    def relative_covariance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""JIT-compiled relative covariance τ^π."""
        jnp = self._jnp
        result = self._jit_fns["relative_covariance"](jnp.array(a), jnp.array(pi))
        return np.asarray(result, dtype=np.float64)

    def portfolio_variance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> float:
        """JIT-compiled portfolio variance π'aπ."""
        jnp = self._jnp
        result = self._jit_fns["portfolio_variance"](jnp.array(a), jnp.array(pi))
        return float(result)

    def simulate_gbm_step(
        self,
        x: NDArray[np.float64],
        mu: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """JIT-compiled exact GBM step."""
        jnp = self._jnp
        result = self._jit_fns["simulate_gbm_step"](
            jnp.array(x),
            jnp.array(mu),
            jnp.array(cholesky),
            jnp.array(dt),
            jnp.array(dw),
        )
        return np.asarray(result, dtype=np.float64)

    def diversity_weights(
        self,
        mu: NDArray[np.float64],
        p: float,
    ) -> NDArray[np.float64]:
        """JIT-compiled diversity weights."""
        jnp = self._jnp
        result = self._jit_fns["diversity_weights"](jnp.array(mu), jnp.array(p))
        return np.asarray(result, dtype=np.float64)

    def covariance_shrinkage(
        self,
        returns: NDArray[np.float64],
        shrinkage: float,
    ) -> NDArray[np.float64]:
        """Shrinkage covariance using JAX."""
        jnp = self._jnp
        jax = self._jax

        @jax.jit  # type: ignore[misc, untyped-decorator]
        def _shrink(rets: Any, alpha: Any) -> Any:  # pragma: no cover
            n_obs = rets.shape[0]
            mean = jnp.mean(rets, axis=0)
            centered = rets - mean
            sample_cov = (centered.T @ centered) / (n_obs - 1)
            n_assets = rets.shape[1]
            target = jnp.eye(n_assets) * jnp.trace(sample_cov) / n_assets
            return (1.0 - alpha) * sample_cov + alpha * target

        result = _shrink(jnp.array(returns), jnp.array(shrinkage))
        return np.asarray(result, dtype=np.float64)

    def gradient(
        self,
        fn: Any,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradient of a scalar function using JAX autodiff.

        Parameters
        ----------
        fn : callable
            Scalar-valued function f(x) -> float.
        x : ndarray
            Point at which to evaluate the gradient.

        Returns
        -------
        ndarray
            Gradient ∇f(x).
        """
        jax = self._jax
        jnp = self._jnp
        grad_fn = jax.grad(fn)
        result = grad_fn(jnp.array(x))
        return np.asarray(result, dtype=np.float64)

    def hessian(
        self,
        fn: Any,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute Hessian of a scalar function using JAX autodiff.

        Parameters
        ----------
        fn : callable
            Scalar-valued function f(x) -> float.
        x : ndarray
            Point at which to evaluate the Hessian.

        Returns
        -------
        ndarray of shape (n, n)
            Hessian matrix ∇²f(x).
        """
        jax = self._jax
        jnp = self._jnp
        hess_fn = jax.hessian(fn)
        result = hess_fn(jnp.array(x))
        return np.asarray(result, dtype=np.float64)
