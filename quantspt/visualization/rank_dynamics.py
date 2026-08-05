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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_rank_changes",
    "plot_rank_transition_heatmap",
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


def plot_rank_changes(
    rank_paths: NDArray[np.intp],
    ax: Axes | None = None,
    n_assets: int | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    r"""Spaghetti plot of rank changes over time.

    Each line traces the rank of a single stock across time steps,
    revealing rank stability and crossing patterns.

    Parameters
    ----------
    rank_paths : ndarray of shape (T, n)
        Rank of each asset at each time step (0 = largest).
    ax : matplotlib Axes, optional
        Axes to plot on.
    n_assets : int, optional
        Number of assets to plot (default: all).
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    BFK §3
    """
    plt = _get_matplotlib()
    require(rank_paths.ndim == 2, "rank_paths must be 2-D (T, n)")

    _n_time, total_assets = rank_paths.shape
    if n_assets is None:
        n_assets = total_assets
    n_assets = min(n_assets, total_assets)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    plot_kwargs: dict[str, Any] = {"alpha": 0.5, "linewidth": 0.8}
    plot_kwargs.update(kwargs)

    for i in range(n_assets):
        ax.plot(rank_paths[:, i], **plot_kwargs)

    ax.set_xlabel("Time")
    ax.set_ylabel("Rank (0 = largest)")
    ax.set_title(title or "Rank Dynamics")
    ax.invert_yaxis()

    return fig


def plot_rank_transition_heatmap(
    transition_matrix: NDArray[np.float64],
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    **kwargs: Any,
) -> Figure:
    r"""Heatmap of the rank transition matrix.

    Entry (i, j) shows the probability of moving from rank i to
    rank j over one time period.

    Parameters
    ----------
    transition_matrix : ndarray of shape (n, n)
        Row-stochastic transition matrix.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    cmap : str
        Colormap name.
    **kwargs
        Passed to ``ax.imshow()``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    BFK Prop. 2.3
    """
    plt = _get_matplotlib()
    require(transition_matrix.ndim == 2, "transition_matrix must be 2-D")
    n = transition_matrix.shape[0]
    require(
        transition_matrix.shape == (n, n),
        f"transition_matrix must be square, got {transition_matrix.shape}",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    im = ax.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1, **kwargs)
    fig.colorbar(im, ax=ax, label="Transition probability")

    ax.set_xlabel("Destination rank")
    ax.set_ylabel("Source rank")
    ax.set_title(title or "Rank Transition Heatmap")

    return fig
