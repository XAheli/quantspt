"""Community contributions and plugin system.

Extensions are discovered via three layers:

1. **Runtime registration** — ``@register_generating_function('my_gf')``
2. **Entry points** — third-party packages declare ``[project.entry-points."quantspt_*"]``
3. **Cookiecutter template** — scaffold a new provider/strategy package

Submodules
----------
registry
    Protocol-based registry and entry-point discovery.
template
    Cookiecutter template for community providers.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_PROVIDER_REGISTRY: dict[str, Any] = {}
_PORTFOLIO_REGISTRY: dict[str, Any] = {}
_MODEL_REGISTRY: dict[str, Any] = {}


def register_data_provider(name: str) -> Callable[[type], type]:
    """Decorator to register a data provider in the runtime registry."""

    def decorator(cls: type) -> type:
        _PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


def register_portfolio(name: str) -> Callable[[type], type]:
    """Decorator to register a portfolio generator."""

    def decorator(cls: type) -> type:
        _PORTFOLIO_REGISTRY[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[type], type]:
    """Decorator to register a market model."""

    def decorator(cls: type) -> type:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def discover_providers() -> dict[str, type]:
    """Discover all installed data providers via entry points."""
    providers = dict(_PROVIDER_REGISTRY)
    for ep in entry_points(group="quantspt_data_provider"):
        providers[ep.name] = ep.load()
    return providers


def discover_portfolios() -> dict[str, type]:
    """Discover all installed portfolio generators via entry points."""
    portfolios = dict(_PORTFOLIO_REGISTRY)
    for ep in entry_points(group="quantspt_portfolio"):
        portfolios[ep.name] = ep.load()
    return portfolios


def discover_models() -> dict[str, type]:
    """Discover all installed market models via entry points."""
    models = dict(_MODEL_REGISTRY)
    for ep in entry_points(group="quantspt_model"):
        models[ep.name] = ep.load()
    return models


def list_providers() -> set[str]:
    """Return names of all discoverable data providers."""
    return set(discover_providers().keys())


def list_portfolios() -> set[str]:
    """Return names of all discoverable portfolio generators."""
    return set(discover_portfolios().keys())


def list_models() -> set[str]:
    """Return names of all discoverable market models."""
    return set(discover_models().keys())


__all__ = [
    "discover_models",
    "discover_portfolios",
    "discover_providers",
    "list_models",
    "list_portfolios",
    "list_providers",
    "register_data_provider",
    "register_model",
    "register_portfolio",
]
