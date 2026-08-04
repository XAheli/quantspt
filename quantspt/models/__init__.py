"""Abstract market models (continuous-time).

All models implement the ``StochasticProcess`` protocol, making them
consumable by any simulator without the simulator knowing the model type.

Submodules
----------
base
    ``MarketModel`` protocol — the common interface for all models.
gbm
    Correlated Geometric Brownian Motion (baseline model).
atlas
    Atlas model (Banner, Fernholz & Karatzas, 2005).
first_order
    General first-order rank-based models (BFK Eq. 1.6).
volatility_stabilized
    Volatility-stabilised market (Lukacs §12, F&K §14).
diverse_market
    Log-pole repulsion models (FKK Theorem 6.1).
hybrid
    Regime-switching and mixture models.
"""

__all__: list[str] = []
