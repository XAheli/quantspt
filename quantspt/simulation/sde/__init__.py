"""Pluggable SDE discretisation schemes.

The same ``StochasticProcess`` can be simulated with different numerical
schemes by swapping the ``Discretization`` strategy.

Submodules
----------
euler_maruyama
    O(√dt) strong convergence, with adaptive step-size.
milstein
    O(dt) strong convergence (scalar SDEs), with adaptive step-size.
"""

from .euler_maruyama import (
    EulerMaruyamaDiscretization,
    adaptive_euler_maruyama,
    verify_convergence_order,
)
from .milstein import (
    MilsteinDiscretization,
    adaptive_milstein,
    verify_milstein_convergence,
)

__all__ = [
    "EulerMaruyamaDiscretization",
    "MilsteinDiscretization",
    "adaptive_euler_maruyama",
    "adaptive_milstein",
    "verify_convergence_order",
    "verify_milstein_convergence",
]
