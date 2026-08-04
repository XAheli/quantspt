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
    Weight evolution and treemaps.
performance
    Cumulative returns, drawdowns, and attribution charts.
model_diagnostics
    QQ-plots, convergence analysis, residual plots.
interactive
    Plotly-based interactive versions.
export
    LaTeX, PDF, and Excel/Jupyter report generation.
"""

__all__: list[str] = []
