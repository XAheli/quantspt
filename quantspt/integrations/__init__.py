"""Integration bridges with external ML/data frameworks.

Submodules
----------
sklearn
    Scikit-learn compatible transformers for SPT features.
vectorbt
    Adapter for vectorbt and backtrader strategy execution.
"""

from __future__ import annotations

__all__: list[str] = []


def __getattr__(name: str) -> object:
    """Lazy-load integration submodules."""
    if name == "SPTSignalFactory":
        from .vectorbt import SPTSignalFactory

        return SPTSignalFactory
    if name == "SPTBacktraderStrategy":
        from .vectorbt import SPTBacktraderStrategy

        return SPTBacktraderStrategy
    raise AttributeError(f"module 'quantspt.integrations' has no attribute {name!r}")
