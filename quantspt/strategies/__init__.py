"""Direct optimization strategies for portfolio construction.

This module provides strategies that target portfolio-level objectives
(excess growth rate, diversification ratio, etc.) directly, without
committing to a generating function. The key advantage: no boundary
term, no structural bet on market concentration.

The flagship strategy is :class:`GammaGradientStrategy`, which targets
the excess growth rate γ* — the mathematical definition of the
diversification return earned through rebalancing.

Classes
-------
GammaGradientStrategy
    Direct γ* gradient targeting. The RECOMMENDED strategy for capturing
    the volatility harvesting premium with controlled risk.
Strategy
    Base protocol for all direct optimization strategies.

Example
-------
>>> from quantspt.strategies import GammaGradientStrategy
>>> import numpy as np
>>> strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.05)
>>> mu = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
>>> cov = np.diag([0.04, 0.06, 0.08, 0.10, 0.12])
>>> weights = strategy.compute_weights(mu, cov)
>>> weights.sum()  # doctest: +ELLIPSIS
1.0...
"""

from quantspt.strategies.base import Strategy, WeightFunction
from quantspt.strategies.gamma_gradient import (
    GammaBacktestResult,
    GammaGradientStrategy,
)
from quantspt.strategies.projections import (
    project_bounded_simplex,
    project_simplex,
)

__all__ = [
    "GammaBacktestResult",
    "GammaGradientStrategy",
    "Strategy",
    "WeightFunction",
    "project_bounded_simplex",
    "project_simplex",
]
