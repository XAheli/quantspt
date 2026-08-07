"""Monte Carlo and SDE simulation engine.

Core design principle: *dynamics are objects*. An SDE is a type with
``drift``/``diffusion``/``evolve``; any simulator can consume it through the
``StochasticProcess`` protocol.

Submodules
----------
sde
    Pluggable discretisation schemes (Euler-Maruyama, Milstein, exact).
monte_carlo
    Monte Carlo framework with variance reduction and CI.
market_simulator
    Simulate from any ``MarketModel``: prices, weights, ranks.
brownian_bridge
    Path-level Brownian bridge construction (binary-tree refinement).
"""

from .brownian_bridge import BrownianBridge
from .market_simulator import MarketSimulation, simulate_market
from .monte_carlo import MonteCarloEngine, MonteCarloResult

__all__ = [
    "BrownianBridge",
    "MarketSimulation",
    "MonteCarloEngine",
    "MonteCarloResult",
    "simulate_market",
]
