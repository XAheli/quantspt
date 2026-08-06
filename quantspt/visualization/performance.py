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
from ._backend import BackendType, _get_matplotlib, _get_plotly, _validate_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_cumulative_returns",
    "plot_master_formula_decomposition",
    "plot_relative_performance",
]


def plot_cumulative_returns(
    returns_dict: dict[str, NDArray[np.float64]],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot cumulative returns for multiple strategies on the same axes.

    Parameters
    ----------
    returns_dict : dict mapping strategy name to return series
        Each value is an ndarray of shape (T,) with simple or log returns.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(len(returns_dict) > 0, "returns_dict must not be empty")
    plot_title = title or "Cumulative Returns"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        for name, rets in returns_dict.items():
            cumulative = np.cumprod(1.0 + rets)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cumulative))),
                    y=cumulative.tolist(),
                    mode="lines",
                    name=name,
                    hovertemplate="%{fullData.name}<br>Time: %{x}<br>Value: %{y:.4f}",
                    **kwargs,
                )
            )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Cumulative return",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        for name, rets in returns_dict.items():
            cumulative = np.cumprod(1.0 + rets)
            ax.plot(cumulative, label=name, **kwargs)

        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative return")
        ax.set_title(plot_title)
        ax.legend(fontsize="small")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        return fig


def plot_relative_performance(
    pi_returns: NDArray[np.float64],
    mu_returns: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Plot relative performance of portfolio vs market.

    Shows the cumulative ratio V_pi / V_mu over time.

    Parameters
    ----------
    pi_returns : ndarray of shape (T,)
        Portfolio returns.
    mu_returns : ndarray of shape (T,)
        Market returns.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(len(pi_returns) == len(mu_returns), "Return series must match in length")

    cum_pi = np.cumprod(1.0 + pi_returns)
    cum_mu = np.cumprod(1.0 + mu_returns)
    relative = cum_pi / cum_mu
    plot_title = title or "Relative Performance"
    time_axis = list(range(len(relative)))

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=relative.tolist(),
                mode="lines",
                name="Portfolio / Market",
                fill="tozeroy",
                hovertemplate="Time: %{x}<br>Relative: %{y:.4f}",
                **kwargs,
            )
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Relative value (portfolio / market)",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        ax.plot(relative, **kwargs)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Relative value (portfolio / market)")
        ax.set_title(plot_title)
        ax.fill_between(
            range(len(relative)),
            1.0,
            relative,
            alpha=0.2,
            where=(relative >= 1.0).tolist(),
            color="green",
        )
        ax.fill_between(
            range(len(relative)),
            1.0,
            relative,
            alpha=0.2,
            where=(relative < 1.0).tolist(),
            color="red",
        )
        return fig


def plot_master_formula_decomposition(
    decomp_dict: dict[str, NDArray[np.float64]],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Plot master formula decomposition: boundary vs drift terms.

    Visualises the components of the master formula that explains
    portfolio outperformance in SPT.

    Parameters
    ----------
    decomp_dict : dict
        Must contain keys ``'boundary'`` and ``'drift'``, each an
        ndarray of shape (T,) with the cumulative contribution.
        May optionally contain ``'residual'``.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying plot call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    F&K Survey Eq. 1.20
    """
    backend = _validate_backend(backend)
    require("boundary" in decomp_dict, "decomp_dict must contain 'boundary'")
    require("drift" in decomp_dict, "decomp_dict must contain 'drift'")

    boundary = decomp_dict["boundary"]
    drift = decomp_dict["drift"]
    total = boundary + drift
    plot_title = title or "Master Formula Decomposition"
    time_axis = list(range(len(boundary)))

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=boundary.tolist(),
                mode="lines",
                name="Boundary term",
                line={"dash": "dash"},
                **kwargs,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=drift.tolist(),
                mode="lines",
                name="Drift term",
                line={"dash": "dashdot"},
                **kwargs,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=total.tolist(),
                mode="lines",
                name="Total",
                line={"width": 3},
                **kwargs,
            )
        )
        if "residual" in decomp_dict:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=decomp_dict["residual"].tolist(),
                    mode="lines",
                    name="Residual",
                    line={"dash": "dot"},
                    opacity=0.6,
                    **kwargs,
                )
            )
        fig.add_hline(y=0.0, line_color="gray", opacity=0.5)
        fig.update_layout(
            title=plot_title,
            xaxis_title="Time",
            yaxis_title="Cumulative contribution",
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

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
        ax.set_title(plot_title)
        ax.legend(fontsize="small")
        return fig
