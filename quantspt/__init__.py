"""quantspt — The Definitive Library for Stochastic Portfolio Theory.

Implements the full mathematical apparatus of Stochastic Portfolio Theory
as developed by E. Robert Fernholz, Ioannis Karatzas, and collaborators.

Quick start::

    import quantspt as spt

    model = spt.FirstOrderModel(n=500, gamma=0.05, g=g, sigma=sigma)
    gen = spt.DiversityGenerator(p=0.76)
    weights = gen.weights(market_weights)
    result = spt.optimize_growth_rate(growth_rates, cov)

Top-level exports follow a *curated noun* pattern: model, generator, and
optimiser objects are available directly; estimators and internals live in
submodules.
"""

from __future__ import annotations

from quantspt._config import get_config, set_backend
from quantspt._result import SPTResult
from quantspt._version import __version__
from quantspt.core.generating_functions import (
    CustomGenerator,
    DiversityGenerator,
    EntropyGenerator,
    GeneratingFunction,
    ModifiedEntropyGenerator,
)
from quantspt.core.processes import CorrelatedGBM
from quantspt.errors import (
    CalibrationError,
    DataProviderError,
    DiversityConditionError,
    OptimizationError,
    SimulationDivergenceError,
    SPTError,
    SPTInvariantError,
)
from quantspt.models import (
    AtlasModel,
    CorrelatedGBMMarket,
    FirstOrderModel,
    MarketModel,
    VolatilityStabilizedMarket,
)
from quantspt.optimization import ConstraintSet, optimize_growth_rate

__all__ = [
    "AtlasModel",
    "CalibrationError",
    "ConstraintSet",
    "CorrelatedGBM",
    "CorrelatedGBMMarket",
    "CustomGenerator",
    "DataProviderError",
    "DiversityConditionError",
    "DiversityGenerator",
    "EntropyGenerator",
    "FirstOrderModel",
    "GeneratingFunction",
    "MarketModel",
    "ModifiedEntropyGenerator",
    "OptimizationError",
    "SPTError",
    "SPTInvariantError",
    "SPTResult",
    "SimulationDivergenceError",
    "VolatilityStabilizedMarket",
    "__version__",
    "get_config",
    "optimize_growth_rate",
    "set_backend",
]
