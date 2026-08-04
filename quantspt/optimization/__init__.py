"""Portfolio optimisation for SPT strategies.

Provides growth-rate maximisation (the correct SPT objective), generating
function parameter tuning, robust optimisation, and constraint construction.

Submodules
----------
growth_rate
    Maximise γ_π using CVXPY (correct SPT formulation).
generating_function
    Optimal G selection and parameter tuning.
robust
    Parameter uncertainty, Bayesian, and minimax approaches.
constraints
    Turnover, cardinality, sector, and position limit builders.
transaction_costs
    Proportional, market impact, and Almgren-Chriss models.
solver
    Solver waterfall (SCS → ECOS → OSQP) with tunable parameters.
multi_period
    Multi-period dynamic optimisation.
"""

__all__: list[str] = []
