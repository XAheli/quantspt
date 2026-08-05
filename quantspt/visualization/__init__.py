"""Publication-quality visualisation for SPT analysis.

All plot functions are *free functions* with ``ax=`` injection,
making them composable with any matplotlib workflow.

Submodules
----------
capital_distribution
    Log-log capital distribution curves.
rank_dynamics
    Rank changes over time, heatmaps.
portfolio_weights
    Weight evolution and strategy comparison.
performance
    Cumulative returns, relative performance, and attribution.
"""

from .capital_distribution import (
    plot_capital_distribution,
    plot_capital_distribution_evolution,
)
from .performance import (
    plot_cumulative_returns,
    plot_master_formula_decomposition,
    plot_relative_performance,
)
from .portfolio_weights import plot_weight_comparison, plot_weight_evolution
from .rank_dynamics import plot_rank_changes, plot_rank_transition_heatmap

__all__ = [
    "plot_capital_distribution",
    "plot_capital_distribution_evolution",
    "plot_cumulative_returns",
    "plot_master_formula_decomposition",
    "plot_rank_changes",
    "plot_rank_transition_heatmap",
    "plot_relative_performance",
    "plot_weight_comparison",
    "plot_weight_evolution",
]
