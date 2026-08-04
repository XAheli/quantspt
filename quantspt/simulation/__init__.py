"""Monte Carlo and SDE simulation engine.

Core design principle: *dynamics are objects*. An SDE is a type with
``drift``/``diffusion``/``evolve``; any simulator can consume it through the
``StochasticProcess`` protocol.

Submodules
----------
sde
    Pluggable discretisation schemes (Euler-Maruyama, Milstein, exact).
path_generator
    Composable MC stack: PathGenerator → PathPricer → Accumulator.
market_simulator
    Simulate from any ``StochasticProcess``.
monte_carlo
    Monte Carlo framework with variance reduction and CI.
antithetic
    Antithetic variates for variance reduction.
importance_sampling
    Rare event simulation.
"""

__all__: list[str] = []
