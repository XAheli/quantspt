"""Relative arbitrage theory.

Implements diversity-based arbitrage detection, mirror portfolios,
minimum horizon computation, and explicit arbitrage construction
from FKK (2005) and the Lukacs Lectures (2006).

Submodules
----------
conditions
    Diversity / weak diversity / asymptotic conditions.
detection
    Arbitrage opportunity screening.
horizon
    Minimum horizon computation (FKK Eq. 4.5).
mirror
    Mirror portfolios (FKK §8).
deflators
    Strict local martingales and EMM failure detection.
construction
    Explicit arbitrage portfolio construction.
"""

__all__: list[str] = []
