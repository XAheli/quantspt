"""Abstract market models (continuous-time).

All models implement the :class:`MarketModel` protocol, making them
consumable by any simulator without the simulator knowing the model type.

Submodules
----------
base
    :class:`MarketModel` — the abstract base for all models.
gbm
    :class:`CorrelatedGBMMarket` — Correlated Geometric Brownian Motion.
atlas
    :class:`AtlasModel`, :class:`FirstOrderModel` — Atlas model
    (Banner, Fernholz & Karatzas, 2005).
volatility_stabilized
    :class:`VolatilityStabilizedMarket` — Volatility-stabilised market
    (Lukacs §12, F&K Survey §14).
"""

from .atlas import AtlasModel, FirstOrderModel
from .base import MarketModel
from .gbm import CorrelatedGBMMarket
from .volatility_stabilized import VolatilityStabilizedMarket

__all__ = [
    "AtlasModel",
    "CorrelatedGBMMarket",
    "FirstOrderModel",
    "MarketModel",
    "VolatilityStabilizedMarket",
]
