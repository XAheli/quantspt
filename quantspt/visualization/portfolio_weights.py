"""Portfolio weight visualisation: evolution and comparison.

Provides plots for tracking weight evolution over time and
comparing weights across different strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from ._backend import BackendType, _get_matplotlib, _get_plotly, _validate_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_weight_comparison",
    "plot_weight_evolution",
]


def plot_weight_evolution(
    weight_paths: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    labels: list[str] | None = None,
    ax: Axes | None = None,
    stacked: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot weight evolution over time.

    Shows how portfolio weights change across time steps, either
    as individual lines or as a stacked area chart.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Weight paths over T time steps for n assets.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    labels : list of str, optional
        Asset labels (default: ``Asset 0``, ``Asset 1``, ...).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    stacked : bool
        If ``True``, use a stacked area plot.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")

    n_time, n_assets = weight_paths.shape
    if labels is None:
        labels = [f"Asset {i}" for i in range(n_assets)]

    plot_title = title or "Weight Evolution"
    time_axis = list(range(n_time))

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        if stacked:
            for i in range(n_assets):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=weight_paths[:, i].tolist(),
                        mode="lines",
                        name=labels[i],
                        stackgroup="weights",
                        hovertemplate="%{fullData.name}: %{y:.4f}",
                        **kwargs,
                    )
                )
        else:
            for i in range(n_assets):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=weight_paths[:, i].tolist(),
                        mode="lines",
                        name=labels[i],
                        hovertemplate="%{fullData.name}: %{y:.4f}",
                        **kwargs,
                    )
                )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Weight",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
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
        ax.set_title(plot_title)
        ax.legend(fontsize="small", loc="upper right")
        return fig


def plot_weight_comparison(
    weights_dict: dict[str, NDArray[np.float64]],
    *,
    backend: BackendType = "plotly",
    tickers: list[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Bar chart comparing portfolio weights across strategies.

    Parameters
    ----------
    weights_dict : dict mapping strategy name to weight vector
        Each value is an ndarray of shape (n,).
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    tickers : list of str, optional
        Asset labels for the x-axis.
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying bar call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(len(weights_dict) > 0, "weights_dict must not be empty")

    first_w = next(iter(weights_dict.values()))
    n_assets = len(first_w)

    if tickers is None:
        tickers = [str(i) for i in range(n_assets)]

    plot_title = title or "Weight Comparison"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for name, w in weights_dict.items():
            fig.add_trace(
                go.Bar(
                    x=tickers,
                    y=w.tolist(),
                    name=name,
                    hovertemplate="%{x}: %{y:.4f}",
                    **kwargs,
                )
            )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Asset",
            yaxis_title="Weight",
            barmode="group",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        strategy_names = list(weights_dict.keys())
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
        ax.set_title(plot_title)
        ax.legend(fontsize="small")
        return fig
