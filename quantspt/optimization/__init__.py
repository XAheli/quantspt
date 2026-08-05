"""Portfolio optimisation for SPT strategies.

Provides growth-rate maximisation (the correct SPT objective), generating
function parameter tuning, and constraint construction.

Submodules
----------
growth_rate
    Maximise portfolio growth rate using CVXPY.
constraints
    Turnover, position limit, and sector constraint builders.
transaction_costs
    Proportional and market impact cost models.
generating_function
    Optimal generating function parameter selection.
"""

from .constraints import (
    ConstraintSet,
    position_limit_constraints,
    sector_constraints,
    turnover_constraint,
)
from .generating_function import (
    OptimizationResult,
    optimize_diversity_parameter,
    optimize_generator_parameter,
)
from .growth_rate import optimize_growth_rate
from .transaction_costs import net_growth_rate, proportional_cost, sqrt_market_impact

__all__ = [
    "ConstraintSet",
    "OptimizationResult",
    "net_growth_rate",
    "optimize_diversity_parameter",
    "optimize_generator_parameter",
    "optimize_growth_rate",
    "position_limit_constraints",
    "proportional_cost",
    "sector_constraints",
    "sqrt_market_impact",
    "turnover_constraint",
]
