"""Portfolio weight visualisation: evolution and comparison.

Provides plots for tracking weight evolution over time and
comparing weights across different strategies.
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
    "plot_weight_comparison",
    "plot_weight_evolution",
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


def plot_weight_evolution(
    weight_paths: NDArray[np.float64],
    labels: list[str] | None = None,
    ax: Axes | None = None,
    stacked: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot weight evolution over time.

    Shows how portfolio weights change across time steps, either
    as individual lines or as a stacked area chart.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Weight paths over T time steps for n assets.
    labels : list of str, optional
        Asset labels (default: ``Asset 0``, ``Asset 1``, ...).
    ax : matplotlib Axes, optional
        Axes to plot on.
    stacked : bool
        If ``True``, use a stacked area plot.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_matplotlib()
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")

    n_time, n_assets = weight_paths.shape
    if labels is None:
        labels = [f"Asset {i}" for i in range(n_assets)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    if stacked:
        ax.stackplot(range(n_time), weight_paths.T, labels=labels, **kwargs)
    else:
        for i in range(n_assets):
            ax.plot(weight_paths[:, i], label=labels[i], **kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Weight")
    ax.set_title(title or "Weight Evolution")
    ax.legend(fontsize="small", loc="upper right")

    return fig


def plot_weight_comparison(
    weights_dict: dict[str, NDArray[np.float64]],
    tickers: list[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Bar chart comparing portfolio weights across strategies.

    Parameters
    ----------
    weights_dict : dict mapping strategy name to weight vector
        Each value is an ndarray of shape (n,).
    tickers : list of str, optional
        Asset labels for the x-axis.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.bar()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_matplotlib()
    require(len(weights_dict) > 0, "weights_dict must not be empty")

    strategy_names = list(weights_dict.keys())
    first_w = next(iter(weights_dict.values()))
    n_assets = len(first_w)

    if tickers is None:
        tickers = [str(i) for i in range(n_assets)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    n_strategies = len(strategy_names)
    bar_width = 0.8 / n_strategies
    x = np.arange(n_assets, dtype=float)

    for idx, name in enumerate(strategy_names):
        offset = (idx - n_strategies / 2 + 0.5) * bar_width
        ax.bar(x + offset, weights_dict[name], bar_width, label=name, **kwargs)

    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_title(title or "Weight Comparison")
    ax.legend(fontsize="small")

    return fig
