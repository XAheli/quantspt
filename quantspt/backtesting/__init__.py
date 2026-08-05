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
statistical_tests
    Bootstrap, permutation, and multiple testing corrections.
"""

from quantspt.backtesting.attribution import AttributionResult, compute_attribution
from quantspt.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantspt.backtesting.execution import (
    ExecutionModel,
    ExecutionResult,
    InstantExecution,
    MarketImpactExecution,
    ProportionalCostExecution,
)
from quantspt.backtesting.performance import (
    PerformanceMetrics,
    TurnoverStats,
    compute_performance,
    compute_turnover_stats,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    tracking_error,
)
from quantspt.backtesting.rebalancing import (
    CalendarRebalancer,
    DriftRebalancer,
    Frequency,
    Rebalancer,
    ThresholdRebalancer,
)
from quantspt.backtesting.statistical_tests import (
    BootstrapCIResult,
    PermutationTestResult,
    bootstrap_confidence_interval,
    multiple_testing_correction,
    permutation_test,
)

__all__ = [
    "AttributionResult",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BootstrapCIResult",
    "CalendarRebalancer",
    "DriftRebalancer",
    "ExecutionModel",
    "ExecutionResult",
    "Frequency",
    "InstantExecution",
    "MarketImpactExecution",
    "PerformanceMetrics",
    "PermutationTestResult",
    "ProportionalCostExecution",
    "Rebalancer",
    "ThresholdRebalancer",
    "TurnoverStats",
    "bootstrap_confidence_interval",
    "compute_attribution",
    "compute_performance",
    "compute_turnover_stats",
    "information_ratio",
    "max_drawdown",
    "multiple_testing_correction",
    "permutation_test",
    "sharpe_ratio",
    "tracking_error",
]
