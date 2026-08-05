"""Performance visualisation: cumulative returns, relative, attribution.

Provides plotting utilities for strategy comparison and master formula
decomposition analysis.

Mathematical References
-----------------------
- Cumulative return: standard compounding
- Master formula decomposition: F&K Survey Eq. 1.20
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_cumulative_returns",
    "plot_master_formula_decomposition",
    "plot_relative_performance",
]


def _get_matplotlib() -> Any:
    """Lazily import matplotlib, raising a clear error if absent."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install quantspt[viz]"
        ) from None


def plot_cumulative_returns(
    returns_dict: dict[str, NDArray[np.float64]],
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot cumulative returns for multiple strategies on the same axes.

    Parameters
    ----------
    returns_dict : dict mapping strategy name to return series
        Each value is an ndarray of shape (T,) with simple or log returns.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_matplotlib()
    require(len(returns_dict) > 0, "returns_dict must not be empty")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for name, rets in returns_dict.items():
        cumulative = np.cumprod(1.0 + rets)
        ax.plot(cumulative, label=name, **kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative return")
    ax.set_title(title or "Cumulative Returns")
    ax.legend(fontsize="small")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)

    return fig


def plot_relative_performance(
    pi_returns: NDArray[np.float64],
    mu_returns: NDArray[np.float64],
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    r"""Plot relative performance of portfolio vs market.

    Shows the cumulative ratio V_pi / V_mu over time.

    Parameters
    ----------
    pi_returns : ndarray of shape (T,)
        Portfolio returns.
    mu_returns : ndarray of shape (T,)
        Market returns.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_matplotlib()
    require(len(pi_returns) == len(mu_returns), "Return series must match in length")

    cum_pi = np.cumprod(1.0 + pi_returns)
    cum_mu = np.cumprod(1.0 + mu_returns)
    relative = cum_pi / cum_mu

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    ax.plot(relative, **kwargs)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative value (portfolio / market)")
    ax.set_title(title or "Relative Performance")
    ax.fill_between(
        range(len(relative)),
        1.0,
        relative,
        alpha=0.2,
        where=relative >= 1.0,
        color="green",
    )
    ax.fill_between(
        range(len(relative)),
        1.0,
        relative,
        alpha=0.2,
        where=relative < 1.0,
        color="red",
    )

    return fig


def plot_master_formula_decomposition(
    decomp_dict: dict[str, NDArray[np.float64]],
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    r"""Plot master formula decomposition: boundary vs drift terms.

    Visualises the components of the master formula that explains
    portfolio outperformance in SPT.

    Parameters
    ----------
    decomp_dict : dict
        Must contain keys ``'boundary'`` and ``'drift'``, each an
        ndarray of shape (T,) with the cumulative contribution.
        May optionally contain ``'residual'``.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    F&K Survey Eq. 1.20
    """
    plt = _get_matplotlib()
    require("boundary" in decomp_dict, "decomp_dict must contain 'boundary'")
    require("drift" in decomp_dict, "decomp_dict must contain 'drift'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    boundary = decomp_dict["boundary"]
    drift = decomp_dict["drift"]
    total = boundary + drift

    ax.plot(boundary, label="Boundary term", linestyle="--", **kwargs)
    ax.plot(drift, label="Drift term", linestyle="-.", **kwargs)
    ax.plot(total, label="Total", linewidth=2, **kwargs)

    if "residual" in decomp_dict:
        ax.plot(
            decomp_dict["residual"],
            label="Residual",
            linestyle=":",
            alpha=0.6,
            **kwargs,
        )

    ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative contribution")
    ax.set_title(title or "Master Formula Decomposition")
    ax.legend(fontsize="small")

    return fig
