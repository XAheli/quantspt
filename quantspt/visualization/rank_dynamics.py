"""Rank dynamics visualisation: spaghetti plots and heatmaps.

Visualises how stock ranks change over time and the structure of
rank transition probabilities.

Mathematical References
-----------------------
- Rank dynamics: BFK §3
- Transition matrix: BFK Prop. 2.3
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
    "plot_rank_changes",
    "plot_rank_transition_heatmap",
]


def plot_rank_changes(
    rank_paths: NDArray[np.intp],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    n_assets: int | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Spaghetti plot of rank changes over time.

    Each line traces the rank of a single stock across time steps,
    revealing rank stability and crossing patterns.

    Parameters
    ----------
    rank_paths : ndarray of shape (T, n)
        Rank of each asset at each time step (0 = largest).
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    n_assets : int, optional
        Number of assets to plot (default: all).
    labels : list of str, optional
        Asset names for legend/hover.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    BFK §3
    """
    backend = _validate_backend(backend)
    require(rank_paths.ndim == 2, "rank_paths must be 2-D (T, n)")

    n_time, total_assets = rank_paths.shape
    if n_assets is None:
        n_assets = total_assets
    n_assets = min(n_assets, total_assets)
    plot_title = title or "Rank Dynamics"

    if labels is None:
        labels = [f"Asset {i}" for i in range(n_assets)]

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for i in range(n_assets):
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_time)),
                    y=rank_paths[:, i].tolist(),
                    mode="lines",
                    opacity=0.6,
                    line={"width": 1},
                    name=labels[i],
                    hovertemplate="Time %{x}<br>Rank: %{y}<br>" + labels[i],
                    **kwargs,
                )
            )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Rank (0 = largest)",
            yaxis_autorange="reversed",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        plot_kwargs: dict[str, Any] = {"alpha": 0.5, "linewidth": 0.8}
        plot_kwargs.update(kwargs)

        for i in range(n_assets):
            ax.plot(rank_paths[:, i], label=labels[i], **plot_kwargs)

        ax.set_xlabel("Time")
        ax.set_ylabel("Rank (0 = largest)")
        ax.set_title(plot_title)
        ax.invert_yaxis()
        return fig


def plot_rank_transition_heatmap(
    transition_matrix: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    **kwargs: Any,
) -> Any:
    r"""Heatmap of the rank transition matrix.

    Entry (i, j) shows the probability of moving from rank i to
    rank j over one time period.

    Parameters
    ----------
    transition_matrix : ndarray of shape (n, n)
        Row-stochastic transition matrix.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    cmap : str
        Colormap name.
    **kwargs
        Passed to the underlying heatmap call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    BFK Prop. 2.3
    """
    backend = _validate_backend(backend)
    require(transition_matrix.ndim == 2, "transition_matrix must be 2-D")
    n = transition_matrix.shape[0]
    require(
        transition_matrix.shape == (n, n),
        f"transition_matrix must be square, got {transition_matrix.shape}",
    )
    plot_title = title or "Rank Transition Heatmap"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure(
            data=go.Heatmap(
                z=transition_matrix,
                zmin=0,
                zmax=1,
                colorscale=cmap.capitalize(),
                colorbar={"title": "Probability"},
                hovertemplate=(
                    "From rank %{y} → rank %{x}<br>P = %{z:.4f}<extra></extra>"
                ),
                **kwargs,
            )
        )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Destination rank",
            yaxis_title="Source rank",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.get_figure()

        im = ax.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1, **kwargs)
        fig.colorbar(im, ax=ax, label="Transition probability")

        ax.set_xlabel("Destination rank")
        ax.set_ylabel("Source rank")
        ax.set_title(plot_title)
        return fig
