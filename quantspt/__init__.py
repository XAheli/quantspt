"""quantspt — The Definitive Library for Stochastic Portfolio Theory.

Implements the full mathematical apparatus of Stochastic Portfolio Theory
as developed by E. Robert Fernholz, Ioannis Karatzas, and collaborators.

Quick start::

    import quantspt as spt

    market = spt.MarketModel.from_prices(prices)
    portfolio = spt.DiversityPortfolio(p=0.76)
    weights = portfolio.generate(market)

Top-level exports follow a *curated noun* pattern: portfolio and market
objects are available directly; estimators and internals live in submodules.
"""

from __future__ import annotations

from quantspt._config import get_config, set_backend
from quantspt._result import SPTResult
from quantspt._version import __version__
from quantspt.errors import (
    CalibrationError,
    DataProviderError,
    DiversityConditionError,
    OptimizationError,
    SimulationDivergenceError,
    SPTError,
    SPTInvariantError,
)

__all__ = [
    "__version__",
    # Configuration
    "get_config",
    "set_backend",
    # Result envelope
    "SPTResult",
    # Errors
    "SPTError",
    "SPTInvariantError",
    "DiversityConditionError",
    "OptimizationError",
    "SimulationDivergenceError",
    "CalibrationError",
    "DataProviderError",
]
