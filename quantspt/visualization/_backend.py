"""Backend management for the visualization package.

Provides lazy-importing helpers for plotly and matplotlib,
along with a unified backend dispatch pattern.
"""

from __future__ import annotations

from typing import Any, Literal

BackendType = Literal["plotly", "matplotlib"]


def _get_plotly() -> Any:
    """Lazily import plotly, raising a clear error if absent."""
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install it with: pip install quantspt[viz]"
        ) from None


def _get_matplotlib() -> tuple[Any, Any]:
    """Lazily import matplotlib, returning (pyplot, Figure).

    Returns
    -------
    tuple
        (matplotlib.pyplot module, matplotlib.figure.Figure class)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        return plt, Figure
    except ImportError:
        raise ImportError(
            "matplotlib is required for static visualization. "
            "Install it with: pip install quantspt[viz]"
        ) from None


def _validate_backend(backend: str) -> BackendType:
    """Validate the backend argument."""
    if backend not in ("plotly", "matplotlib"):
        raise ValueError(f"backend must be 'plotly' or 'matplotlib', got '{backend}'")
    return backend  # type: ignore[return-value]
