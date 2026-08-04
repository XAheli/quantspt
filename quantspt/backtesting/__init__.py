"""Historical backtesting engine for SPT strategies.

Features event-driven architecture, realistic execution simulation,
SPT-specific attribution via the master formula, and walk-forward
parameter estimation.

Submodules
----------
engine
    Event-driven backtesting core.
rebalancing
    Calendar, threshold, and drift-based rebalancing triggers.
execution
    Realistic fill simulation with market impact.
performance
    SPT-specific performance metrics and attribution.
attribution
    Master formula performance decomposition.
walk_forward
    Walk-forward optimisation framework.
statistical_tests
    Bootstrap, permutation, and multiple testing corrections.
"""

__all__: list[str] = []
