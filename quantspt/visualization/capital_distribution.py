"""Capital distribution curve visualisation.

The capital distribution curve is the signature plot of SPT: it shows
market weights vs rank on a log-log scale, revealing the power-law
structure of equity markets.

Mathematical References
-----------------------
- Capital distribution: F&K Survey Eq. 1.18
- Log-log Pareto structure: BFK §4
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
    "plot_capital_distribution",
    "plot_capital_distribution_evolution",
]


def plot_capital_distribution(
    weights: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    log_scale: bool = True,
    title: str | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    r"""Plot the capital distribution curve (ranked weights).

    Displays market weights sorted in descending order vs rank,
    typically on a log-log scale to reveal the power-law structure.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Market weights (must sum to 1, non-negative).
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    log_scale : bool
        Whether to use log-log axes (default ``True``).
    title : str, optional
        Plot title.
    labels : list of str, optional
        Asset names for hover text (plotly backend).
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    F&K Survey Eq. 1.18, BFK §4
    """
    backend = _validate_backend(backend)
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")

    sorted_indices = np.argsort(weights)[::-1]
    sorted_w = weights[sorted_indices]
    ranks = np.arange(1, len(sorted_w) + 1)

    if labels is not None:
        sorted_labels = [labels[i] for i in sorted_indices]
    else:
        sorted_labels = [f"Asset {i}" for i in sorted_indices]

    plot_title = title or "Capital Distribution Curve"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ranks,
                y=sorted_w,
                mode="lines+markers",
                marker={"size": 4},
                text=sorted_labels,
                hovertemplate="Rank %{x}<br>Weight: %{y:.6f}<br>%{text}",
                name="Weights",
                **kwargs,
            )
        )
        axis_type = "log" if log_scale else "linear"
        fig.update_layout(
            title=plot_title,
            xaxis_title="Rank",
            yaxis_title="Market weight",
            xaxis_type=axis_type,
            yaxis_type=axis_type,
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        plot_kwargs: dict[str, Any] = {
            "marker": "o",
            "markersize": 3,
            "linewidth": 1,
        }
        plot_kwargs.update(kwargs)
        ax.plot(ranks, sorted_w, **plot_kwargs)

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Market weight")
        ax.set_title(plot_title)
        return fig


def plot_capital_distribution_evolution(
    weight_paths: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    times: NDArray[np.float64] | None = None,
    n_snapshots: int = 5,
    ax: Axes | None = None,
    log_scale: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Plot the evolution of the capital distribution curve over time.

    Shows multiple snapshots of the ranked-weight curve at different
    time points, revealing how market concentration changes.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Market weight paths over T time steps for n assets.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    times : ndarray of shape (T,), optional
        Time labels for each snapshot.
    n_snapshots : int
        Number of evenly-spaced time snapshots to display.
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    log_scale : bool
        Whether to use log-log axes.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    BFK §4
    """
    backend = _validate_backend(backend)
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")

    n_time, n_assets = weight_paths.shape
    indices = np.linspace(0, n_time - 1, n_snapshots, dtype=int)
    ranks = np.arange(1, n_assets + 1)
    plot_title = title or "Capital Distribution Evolution"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for idx in indices:
            sorted_w = np.sort(weight_paths[idx])[::-1]
            label = f"t={times[idx]:.2f}" if times is not None else f"step {idx}"
            fig.add_trace(
                go.Scatter(
                    x=ranks,
                    y=sorted_w,
                    mode="lines+markers",
                    marker={"size": 3},
                    name=label,
                    **kwargs,
                )
            )
        axis_type = "log" if log_scale else "linear"
        fig.update_layout(
            title=plot_title,
            xaxis_title="Rank",
            yaxis_title="Market weight",
            xaxis_type=axis_type,
            yaxis_type=axis_type,
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        for idx in indices:
            sorted_w = np.sort(weight_paths[idx])[::-1]
            label = f"t={times[idx]:.2f}" if times is not None else f"step {idx}"
            ax.plot(ranks, sorted_w, label=label, **kwargs)

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("Rank")
        ax.set_ylabel("Market weight")
        ax.set_title(plot_title)
        ax.legend(fontsize="small")
        return fig
