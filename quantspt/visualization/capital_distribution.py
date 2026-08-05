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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_capital_distribution",
    "plot_capital_distribution_evolution",
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


def plot_capital_distribution(
    weights: NDArray[np.float64],
    ax: Axes | None = None,
    log_scale: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    r"""Plot the capital distribution curve (ranked weights).

    Displays market weights sorted in descending order vs rank,
    typically on a log-log scale to reveal the power-law structure.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Market weights (must sum to 1, non-negative).
    ax : matplotlib Axes, optional
        Axes to plot on.  If ``None``, a new figure is created.
    log_scale : bool
        Whether to use log-log axes (default ``True``).
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    F&K Survey Eq. 1.18, BFK §4
    """
    plt = _get_matplotlib()
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")

    sorted_w = np.sort(weights)[::-1]
    ranks = np.arange(1, len(sorted_w) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    plot_kwargs: dict[str, Any] = {"marker": "o", "markersize": 3, "linewidth": 1}
    plot_kwargs.update(kwargs)
    ax.plot(ranks, sorted_w, **plot_kwargs)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Market weight")
    ax.set_title(title or "Capital Distribution Curve")

    return fig


def plot_capital_distribution_evolution(
    weight_paths: NDArray[np.float64],
    times: NDArray[np.float64] | None = None,
    n_snapshots: int = 5,
    ax: Axes | None = None,
    log_scale: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Figure:
    r"""Plot the evolution of the capital distribution curve over time.

    Shows multiple snapshots of the ranked-weight curve at different
    time points, revealing how market concentration changes.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Market weight paths over T time steps for n assets.
    times : ndarray of shape (T,), optional
        Time labels for each snapshot.
    n_snapshots : int
        Number of evenly-spaced time snapshots to display.
    ax : matplotlib Axes, optional
        Axes to plot on.
    log_scale : bool
        Whether to use log-log axes.
    title : str, optional
        Plot title.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.figure.Figure

    References
    ----------
    BFK §4
    """
    plt = _get_matplotlib()
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")

    n_time, n_assets = weight_paths.shape
    indices = np.linspace(0, n_time - 1, n_snapshots, dtype=int)
    ranks = np.arange(1, n_assets + 1)

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
    ax.set_title(title or "Capital Distribution Evolution")
    ax.legend(fontsize="small")

    return fig
