"""Publication-quality visualisation for SPT analysis.

Dual-backend plotting: Plotly (interactive, default) with matplotlib
fallback (static, for publications).

All plot functions accept ``backend='plotly'`` (default) or
``backend='matplotlib'`` to choose the rendering engine.
Matplotlib functions support ``ax=`` injection for composability.

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
model_diagnostics
    QQ-plots, convergence diagnostics, residuals.
interactive
    3D simplex trajectory, generating function surface, arbitrage cone.
"""

from .capital_distribution import (
    plot_capital_distribution,
    plot_capital_distribution_evolution,
)
from .interactive import (
    plot_arbitrage_horizon_cone,
    plot_generating_function_surface,
    plot_simplex_trajectory,
)
from .model_diagnostics import plot_convergence, plot_qq, plot_residuals
from .performance import (
    plot_cumulative_returns,
    plot_master_formula_decomposition,
    plot_relative_performance,
)
from .portfolio_weights import plot_weight_comparison, plot_weight_evolution
from .rank_dynamics import plot_rank_changes, plot_rank_transition_heatmap

__all__ = [
    "plot_arbitrage_horizon_cone",
    "plot_capital_distribution",
    "plot_capital_distribution_evolution",
    "plot_convergence",
    "plot_cumulative_returns",
    "plot_generating_function_surface",
    "plot_master_formula_decomposition",
    "plot_qq",
    "plot_rank_changes",
    "plot_rank_transition_heatmap",
    "plot_relative_performance",
    "plot_residuals",
    "plot_simplex_trajectory",
    "plot_weight_comparison",
    "plot_weight_evolution",
]
