"""Compute backend abstraction.

Provides a uniform array API across NumPy (default), Numba JIT,
JAX (GPU + autodiff), and CuPy (raw CUDA). The active backend is
selected via :func:`quantspt.set_backend`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Backend",
    "get_backend",
    "list_backends",
    "register_backend",
    "set_backend",
]


@runtime_checkable
class Backend(Protocol):
    """Protocol defining the compute backend interface."""

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    def excess_growth_rate(
        self,
        pi: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> float:
        r"""Compute excess growth rate γ*_π."""
        ...

    def relative_covariance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Compute relative covariance matrix τ^π."""
        ...

    def portfolio_variance(
        self,
        a: NDArray[np.float64],
        pi: NDArray[np.float64],
    ) -> float:
        """Compute π'aπ."""
        ...

    def simulate_gbm_step(
        self,
        x: NDArray[np.float64],
        mu: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Single GBM step via exact log-normal transition."""
        ...

    def diversity_weights(
        self,
        mu: NDArray[np.float64],
        p: float,
    ) -> NDArray[np.float64]:
        """Diversity-weighted portfolio: π_i = μ_i^p / Σ μ_j^p."""
        ...

    def covariance_shrinkage(
        self,
        returns: NDArray[np.float64],
        shrinkage: float,
    ) -> NDArray[np.float64]:
        """Shrinkage covariance estimator."""
        ...


_registry: dict[str, Backend] = {}
_active_backend: str = "numpy"


def register_backend(name: str, backend: Backend) -> None:
    """Register a compute backend.

    Parameters
    ----------
    name : str
        Unique identifier for the backend.
    backend : Backend
        Object implementing the Backend protocol.
    """
    _registry[name] = backend


def get_backend(name: str | None = None) -> Backend:
    """Retrieve a backend by name or the active one.

    Parameters
    ----------
    name : str, optional
        Backend name. If ``None``, returns the active backend.

    Returns
    -------
    Backend
        The requested backend instance.

    Raises
    ------
    KeyError
        If the named backend is not registered.
    """
    target = name or _active_backend
    if target not in _registry:
        if target == "numpy":
            from .numpy_backend import NumpyBackend

            _registry["numpy"] = NumpyBackend()
        elif target == "jax":
            from .jax_backend import JaxBackend

            _registry["jax"] = JaxBackend()
        elif target == "numba":
            from .numba_backend import NumbaBackend

            _registry["numba"] = NumbaBackend()
        else:
            raise KeyError(
                f"Backend '{target}' not registered. "
                f"Available: {list(_registry.keys())}"
            )
    return _registry[target]


def set_backend(name: str) -> None:
    """Set the active compute backend.

    Parameters
    ----------
    name : str
        Backend name. Must be registered or a built-in name.
    """
    global _active_backend
    _ = get_backend(name)
    _active_backend = name


def list_backends() -> list[str]:
    """List all registered backend names."""
    if "numpy" not in _registry:
        from .numpy_backend import NumpyBackend

        _registry["numpy"] = NumpyBackend()
    return list(_registry.keys())


def _reset_registry() -> None:
    """Reset registry to defaults (for testing)."""
    global _active_backend
    _registry.clear()
    _active_backend = "numpy"
