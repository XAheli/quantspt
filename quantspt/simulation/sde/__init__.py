"""Pluggable SDE discretisation schemes.

The same ``StochasticProcess`` can be simulated with different numerical
schemes by swapping the ``Discretization`` strategy.

Submodules
----------
discretization
    ``Discretization`` protocol and scheme base.
euler_maruyama
    O(√dt) strong convergence.
milstein
    O(dt) strong convergence.
exact
    Exact simulation for GBM, OU, and other tractable processes.
adaptive
    Adaptive step-size control for stiff dynamics.
"""

__all__: list[str] = []
