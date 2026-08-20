"""Model diagnostics visualisation: QQ-plots, convergence, residuals.

Provides diagnostic plots for assessing model fit quality,
convergence of estimation procedures, and residual analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .._preconditions import require
from ._backend import BackendType, _get_matplotlib, _get_plotly, _validate_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_convergence",
    "plot_qq",
    "plot_residuals",
]


def plot_qq(
    residuals: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """QQ-plot of residuals against the standard normal distribution.

    Parameters
    ----------
    residuals : ndarray of shape (n,)
        Sample residuals or standardised innovations.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying scatter call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(residuals.ndim == 1, f"residuals must be 1-D, got ndim={residuals.ndim}")
    require(len(residuals) >= 2, "need at least 2 data points for QQ-plot")

    sorted_resid = np.sort(residuals)
    n = len(sorted_resid)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    plot_title = title or "QQ-Plot (Normal)"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles.tolist(),
                y=sorted_resid.tolist(),
                mode="markers",
                marker={"size": 4},
                name="Residuals",
                hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}",
                **kwargs,
            )
        )
        # Reference line
        min_val = min(theoretical_quantiles.min(), sorted_resid.min())
        max_val = max(theoretical_quantiles.max(), sorted_resid.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line={"dash": "dash", "color": "red"},
                name="45° line",
                showlegend=True,
            )
        )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        ax.scatter(theoretical_quantiles, sorted_resid, s=10, **kwargs)
        min_val = min(theoretical_quantiles.min(), sorted_resid.min())
        max_val = max(theoretical_quantiles.max(), sorted_resid.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(plot_title)
        return fig


def plot_convergence(
    loss_history: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    log_scale: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot convergence of an iterative estimation procedure.

    Parameters
    ----------
    loss_history : ndarray of shape (n_iters,)
        Loss or objective value at each iteration.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    log_scale : bool
        Whether to use log y-axis.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(
        loss_history.ndim == 1,
        f"loss_history must be 1-D, got ndim={loss_history.ndim}",
    )

    plot_title = title or "Convergence"
    iterations = list(range(1, len(loss_history) + 1))

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=loss_history.tolist(),
                mode="lines",
                name="Loss",
                hovertemplate="Iter %{x}: %{y:.6f}",
                **kwargs,
            )
        )
        y_type = "log" if log_scale else "linear"
        fig.update_layout(
            title=plot_title,
            xaxis_title="Iteration",
            yaxis_title="Loss",
            yaxis_type=y_type,
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(iterations, loss_history, **kwargs)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(plot_title)
        return fig


def plot_residuals(
    residuals: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Residual plot with zero-line and ±2σ bands.

    Parameters
    ----------
    residuals : ndarray of shape (n,)
        Model residuals.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying scatter call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(residuals.ndim == 1, f"residuals must be 1-D, got ndim={residuals.ndim}")

    plot_title = title or "Residuals"
    sigma = float(np.std(residuals, ddof=1))
    indices = list(range(len(residuals)))

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=residuals.tolist(),
                mode="markers",
                marker={"size": 4},
                name="Residuals",
                hovertemplate="Obs %{x}: %{y:.4f}",
                **kwargs,
            )
        )
        fig.add_hline(y=0, line_color="gray")
        fig.add_hline(
            y=2 * sigma,
            line_dash="dash",
            line_color="orange",
            annotation_text="+2\u03c3",
        )
        fig.add_hline(
            y=-2 * sigma,
            line_dash="dash",
            line_color="orange",
            annotation_text="-2\u03c3",
        )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Observation",
            yaxis_title="Residual",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        ax.scatter(indices, residuals, s=10, **kwargs)
        ax.axhline(y=0, color="gray", linewidth=1)
        ax.axhline(y=2 * sigma, color="orange", linestyle="--", linewidth=0.8)
        ax.axhline(y=-2 * sigma, color="orange", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Residual")
        ax.set_title(plot_title)
        return fig
