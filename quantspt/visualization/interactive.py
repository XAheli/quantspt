"""Interactive 3D visualisation for SPT.

Provides advanced interactive plots including:
- 3D simplex trajectory for 3-asset markets
- Generating function surfaces
- Arbitrage horizon cones

These are inherently interactive and default to plotly only,
with a matplotlib static fallback for publication figures.
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
    "plot_arbitrage_horizon_cone",
    "plot_generating_function_surface",
    "plot_simplex_trajectory",
]


def _barycentric_to_cartesian(
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert 3-simplex barycentric coordinates to 2D Cartesian."""
    x = 0.5 * (2 * weights[:, 1] + weights[:, 2])
    y = (np.sqrt(3.0) / 2.0) * weights[:, 2]
    return x, y


def plot_simplex_trajectory(
    weight_paths: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""3D simplex trajectory for a 3-asset market.

    Projects the weight path onto the 2-simplex (equilateral triangle)
    and optionally shows the time dimension as a 3D z-axis.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, 3)
        Weight paths for exactly 3 assets over T time steps.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only; uses 2D projection).
    labels : list of str, optional
        Names for the 3 vertices.
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying trace call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, 3)")
    require(
        weight_paths.shape[1] == 3,
        f"simplex trajectory requires exactly 3 assets, got {weight_paths.shape[1]}",
    )

    n_time = weight_paths.shape[0]
    plot_title = title or "Simplex Trajectory (3-Asset Market)"
    if labels is None:
        labels = ["Asset 0", "Asset 1", "Asset 2"]

    x, y = _barycentric_to_cartesian(weight_paths)
    time_axis = np.arange(n_time)

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x.tolist(),
                y=y.tolist(),
                z=time_axis.tolist(),
                mode="lines+markers",
                marker={
                    "size": 2,
                    "color": time_axis.tolist(),
                    "colorscale": "Viridis",
                },
                line={"width": 2},
                name="Trajectory",
                hovertemplate=(
                    "w₀=%{customdata[0]:.3f} "
                    "w₁=%{customdata[1]:.3f} "
                    "w₂=%{customdata[2]:.3f}<br>"
                    "t=%{z}"
                ),
                customdata=weight_paths.tolist(),
                **kwargs,
            )
        )
        # Simplex boundary triangle at z=0
        corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        cx, cy = _barycentric_to_cartesian(corners)
        fig.add_trace(
            go.Scatter3d(
                x=cx.tolist(),
                y=cy.tolist(),
                z=[0, 0, 0, 0],
                mode="lines",
                line={"color": "black", "width": 3},
                name="Simplex boundary",
                showlegend=False,
            )
        )
        fig.update_layout(
            title=plot_title,
            scene={
                "xaxis_title": labels[1],
                "yaxis_title": labels[2],
                "zaxis_title": "Time",
            },
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.get_figure()

        # 2D simplex projection
        scatter = ax.scatter(x, y, c=time_axis, cmap="viridis", s=5, **kwargs)
        ax.plot(x, y, alpha=0.3, linewidth=0.5, color="gray")
        fig.colorbar(scatter, ax=ax, label="Time step")

        # Draw simplex boundary
        corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        cx, cy = _barycentric_to_cartesian(corners)
        ax.plot(cx, cy, "k-", linewidth=2)

        # Vertex labels
        vx, vy = _barycentric_to_cartesian(np.eye(3))
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (vx[i], vy[i]), fontsize=10, ha="center")

        ax.set_aspect("equal")
        ax.set_title(plot_title)
        ax.axis("off")
        return fig


def plot_generating_function_surface(
    func: Any,
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    resolution: int = 50,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Surface plot of a generating function over the 2-simplex.

    Evaluates a generating function g(w₀, w₁, w₂) on a grid over
    the 3-asset simplex and displays the resulting surface.

    Parameters
    ----------
    func : callable
        Generating function accepting an ndarray of shape (n, 3)
        and returning an ndarray of shape (n,).
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes for 2D contour (matplotlib backend only).
    resolution : int
        Grid resolution per axis (default 50).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying surface/contour call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    backend = _validate_backend(backend)
    require(callable(func), "func must be callable")
    require(resolution >= 3, f"resolution must be >= 3, got {resolution}")

    # Generate grid points on the simplex
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append([i / resolution, j / resolution, k / resolution])
    points_arr = np.array(points)
    values = np.asarray(func(points_arr), dtype=np.float64)
    x, y = _barycentric_to_cartesian(points_arr)

    plot_title = title or "Generating Function Surface"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        fig.add_trace(
            go.Mesh3d(
                x=x.tolist(),
                y=y.tolist(),
                z=values.tolist(),
                intensity=values.tolist(),
                colorscale="Viridis",
                name="g(w)",
                hovertemplate="g = %{z:.4f}",
                **kwargs,
            )
        )
        fig.update_layout(
            title=plot_title,
            scene={
                "xaxis_title": "w₁ direction",
                "yaxis_title": "w₂ direction",
                "zaxis_title": "g(w)",
            },
        )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.get_figure()

        scatter = ax.tricontourf(x, y, values, levels=20, cmap="viridis", **kwargs)
        fig.colorbar(scatter, ax=ax, label="g(w)")
        ax.set_aspect("equal")
        ax.set_title(plot_title)
        ax.axis("off")
        return fig


def plot_arbitrage_horizon_cone(
    horizon_data: NDArray[np.float64],
    *,
    backend: BackendType = "plotly",
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""Arbitrage horizon cone visualisation.

    Displays the time-varying cone of portfolios that achieve
    relative arbitrage within a given horizon, projected onto
    a chosen subspace.

    Parameters
    ----------
    horizon_data : ndarray of shape (n_horizons, n_points, 2) or (n_horizons, n_points, 3)
        For each horizon level, the boundary points of the arbitrage cone.
        Last dimension: (x, y) for 2D or (x, y, z) for 3D.
    backend : {'plotly', 'matplotlib'}
        Plotting backend (default ``'plotly'``).
    ax : matplotlib Axes, optional
        Axes to plot on (matplotlib backend only; 2D only).
    title : str, optional
        Plot title.
    **kwargs
        Passed to the underlying trace call.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure

    References
    ----------
    FKK, F&K Survey §6
    """
    backend = _validate_backend(backend)
    require(horizon_data.ndim == 3, "horizon_data must be 3-D (horizons, points, dims)")
    n_horizons, _n_points, dims = horizon_data.shape
    require(dims in (2, 3), f"last dimension must be 2 or 3, got {dims}")

    plot_title = title or "Arbitrage Horizon Cone"

    if backend == "plotly":
        go = _get_plotly()
        fig = go.Figure()
        if dims == 3:
            for h in range(n_horizons):
                pts = horizon_data[h]
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0].tolist(),
                        y=pts[:, 1].tolist(),
                        z=pts[:, 2].tolist(),
                        mode="lines",
                        name=f"Horizon {h}",
                        opacity=0.5 + 0.5 * (h / max(1, n_horizons - 1)),
                        **kwargs,
                    )
                )
            fig.update_layout(
                title=plot_title,
                scene={
                    "xaxis_title": "x",
                    "yaxis_title": "y",
                    "zaxis_title": "z",
                },
            )
        else:
            for h in range(n_horizons):
                pts = horizon_data[h]
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0].tolist(),
                        y=pts[:, 1].tolist(),
                        mode="lines",
                        name=f"Horizon {h}",
                        opacity=0.5 + 0.5 * (h / max(1, n_horizons - 1)),
                        **kwargs,
                    )
                )
            fig.update_layout(
                title=plot_title,
                xaxis_title="x",
                yaxis_title="y",
            )
        return fig
    else:
        plt, _ = _get_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        for h in range(n_horizons):
            pts = horizon_data[h]
            alpha = 0.3 + 0.7 * (h / max(1, n_horizons - 1))
            ax.plot(pts[:, 0], pts[:, 1], alpha=alpha, label=f"Horizon {h}", **kwargs)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(plot_title)
        ax.legend(fontsize="small")
        return fig
