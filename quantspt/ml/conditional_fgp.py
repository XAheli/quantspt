"""Re-export from quantspt.experimental.conditional_fgp (canonical location).

Conditional generating functions are a research direction.  For production
use, prefer ``DiversityGenerator`` with ``quantspt.universe.SPTUniverseSelector``.
"""

from quantspt.experimental.conditional_fgp import *  # noqa: F403
from quantspt.experimental.conditional_fgp import (  # explicit re-exports
    DEFAULT_FEATURES,
    P_MAX,
    P_MIN,
    BoundaryRobustnessRegularizer,
    ConditionalFGPConfig,
    CovarianceConditionalFGP,
    CovarianceFeatureExtractor,
    cost_optimal_p,
    optimal_p_for_cost_level,
)

__all__ = [
    "DEFAULT_FEATURES",
    "P_MAX",
    "P_MIN",
    "BoundaryRobustnessRegularizer",
    "ConditionalFGPConfig",
    "CovarianceConditionalFGP",
    "CovarianceFeatureExtractor",
    "cost_optimal_p",
    "optimal_p_for_cost_level",
]
