"""Global configuration for quantspt.

Controls the compute backend, default numerical tolerances, and logging.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


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
