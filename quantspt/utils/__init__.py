"""Shared utilities for quantspt.

Contains numerical helpers, floating-point comparison, and other
cross-cutting concerns.
"""

from __future__ import annotations

import numpy as np


def close(a: float, b: float, n: int = 42) -> bool:
    """Knuth-style floating-point comparison.

    Returns ``True`` when ``|a - b| ≤ ε · max(|a|, |b|, 1)`` where
    ``ε = n × machine_epsilon``.

    Parameters
    ----------
    a, b
        Values to compare.
    n
        Multiplier on machine epsilon (default 42).
    """
    eps = n * np.finfo(np.float64).eps
    return bool(abs(a - b) <= eps * max(abs(a), abs(b), 1.0))


__all__ = ["close"]
