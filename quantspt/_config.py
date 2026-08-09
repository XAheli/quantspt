"""Global configuration for quantspt.

Controls the compute backend, default numerical tolerances, and logging.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import Any, TypeVar

F = TypeVar("F")


@dataclass
class _SPTConfig:
    """Library-wide settings (singleton accessed via :func:`get_config`)."""

    backend: str = "numpy"
    float_tolerance_ulps: int = 42
    default_seed: int | None = None
    strict_validation: bool = True


_GLOBAL_CONFIG = _SPTConfig()
_CONFIG_LOCK = threading.Lock()


def get_config() -> _SPTConfig:
    """Return the global configuration object."""
    return _GLOBAL_CONFIG


def set_backend(name: str) -> None:
    """Select the compute backend: ``'numpy'``, ``'numba'``, ``'jax'``, or ``'cupy'``.

    Thread-safe but intended to be called once at process startup.
    Dynamic switching during computation is not recommended.
    """
    allowed = {"numpy", "numba", "jax", "cupy"}
    if name not in allowed:
        raise ValueError(f"Unknown backend {name!r}. Choose from {allowed}")
    with _CONFIG_LOCK:
        _GLOBAL_CONFIG.backend = name


def backend_dispatch(fn: Any) -> Any:
    """Decorator that routes a core function to the active backend.

    When ``set_backend("jax")`` or ``set_backend("numba")`` has been
    called, the decorated function checks whether the active backend
    object exposes a method with the same name and delegates to it.
    If the backend doesn't implement the method, or the backend is
    ``"numpy"`` (default), the original NumPy implementation runs.

    Apply to core mathematical functions that have equivalent
    implementations in ``_backends.jax_backend`` or
    ``_backends.numba_backend``.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        backend_name = _GLOBAL_CONFIG.backend
        if backend_name != "numpy":
            try:
                from ._backends import get_backend

                be = get_backend(backend_name)
                method = getattr(be, fn.__name__, None)
                if method is not None:
                    return method(*args, **kwargs)
            except (KeyError, ImportError):
                pass
        return fn(*args, **kwargs)

    return wrapper
