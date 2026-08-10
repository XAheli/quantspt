"""SPT-optimised universe selection.

Select stocks that maximise the excess growth rate γ* of a diversity-weighted
portfolio.  The module provides scoring criteria, a composite selector, and
reconstitution logic with hysteresis to limit turnover.

Quick start::

    from quantspt.universe import SPTUniverseSelector

    selector = SPTUniverseSelector(n_stocks=30, max_market_cap_percentile=85)
    universe = selector.select(all_prices)

    # Now run strategy only on selected stocks
    from quantspt import DiversityGenerator
    weights = DiversityGenerator(p=0.3).weights(market_weights[universe])
"""

from .criteria import (
    boundary_risk_score,
    gamma_star_contribution,
    idiosyncratic_volatility,
    liquidity_filter,
    pairwise_correlation_score,
)
from .reconstitution import UniverseReconstitution
from .selector import SPTUniverseSelector

__all__ = [
    "SPTUniverseSelector",
    "UniverseReconstitution",
    "boundary_risk_score",
    "gamma_star_contribution",
    "idiosyncratic_volatility",
    "liquidity_filter",
    "pairwise_correlation_score",
]
