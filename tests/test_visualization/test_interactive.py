"""Tests for visualization/interactive.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.interactive import (
    plot_arbitrage_horizon_cone,
    plot_generating_function_surface,
    plot_simplex_trajectory,
)


@pytest.fixture()
def simplex_paths():
    """Synthetic 3-asset weight paths on the simplex."""
    rng = np.random.default_rng(42)
    paths = np.zeros((30, 3))
    w = rng.dirichlet(np.ones(3))
    for t in range(30):
        noise = rng.standard_normal(3) * 0.02
        w = np.abs(w + noise)
        w /= w.sum()
        paths[t] = w
    return paths


@pytest.fixture()
def horizon_data_2d():
    """Synthetic 2D horizon cone data."""
    n_horizons, n_points = 3, 20
    theta = np.linspace(0, 2 * np.pi, n_points)
    data = np.zeros((n_horizons, n_points, 2))
    for h in range(n_horizons):
        radius = 1.0 + h * 0.5
        data[h, :, 0] = radius * np.cos(theta)
        data[h, :, 1] = radius * np.sin(theta)
    return data


@pytest.fixture()
def horizon_data_3d():
    """Synthetic 3D horizon cone data."""
    n_horizons, n_points = 3, 20
    theta = np.linspace(0, 2 * np.pi, n_points)
    data = np.zeros((n_horizons, n_points, 3))
    for h in range(n_horizons):
        radius = 1.0 + h * 0.5
        data[h, :, 0] = radius * np.cos(theta)
        data[h, :, 1] = radius * np.sin(theta)
        data[h, :, 2] = h * np.ones(n_points)
    return data


def _sample_gen_func(weights: np.ndarray) -> np.ndarray:
    """A simple test generating function: log-diversity."""
    return np.sum(np.log(np.maximum(weights, 1e-10)), axis=1)


class TestSimplexTrajectoryPlotly:
    def test_returns_plotly_figure(self, simplex_paths) -> None:
        fig = plot_simplex_trajectory(simplex_paths, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_with_labels(self, simplex_paths) -> None:
        labels = ["AAPL", "GOOG", "MSFT"]
        fig = plot_simplex_trajectory(simplex_paths, backend="plotly", labels=labels)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, simplex_paths) -> None:
        fig = plot_simplex_trajectory(simplex_paths, backend="plotly", title="Test")
        assert fig.layout.title.text == "Test"

    def test_rejects_non_3_assets(self) -> None:
        from quantspt.errors import SPTInvariantError

        with pytest.raises(SPTInvariantError, match="3 assets"):
            plot_simplex_trajectory(np.ones((10, 4)) / 4, backend="plotly")


class TestSimplexTrajectoryMatplotlib:
    def test_returns_mpl_figure(self, simplex_paths) -> None:
        fig = plot_simplex_trajectory(simplex_paths, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, simplex_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_simplex_trajectory(simplex_paths, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestGeneratingFunctionSurfacePlotly:
    def test_returns_plotly_figure(self) -> None:
        fig = plot_generating_function_surface(_sample_gen_func, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_resolution(self) -> None:
        fig = plot_generating_function_surface(
            _sample_gen_func, backend="plotly", resolution=10
        )
        assert isinstance(fig, go.Figure)

    def test_custom_title(self) -> None:
        fig = plot_generating_function_surface(
            _sample_gen_func, backend="plotly", title="Gen Func"
        )
        assert fig.layout.title.text == "Gen Func"


class TestGeneratingFunctionSurfaceMatplotlib:
    def test_returns_mpl_figure(self) -> None:
        fig = plot_generating_function_surface(_sample_gen_func, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_generating_function_surface(
            _sample_gen_func, backend="matplotlib", ax=ax
        )
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestArbitrageHorizonConePlotly:
    def test_returns_plotly_figure_2d(self, horizon_data_2d) -> None:
        fig = plot_arbitrage_horizon_cone(horizon_data_2d, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_returns_plotly_figure_3d(self, horizon_data_3d) -> None:
        fig = plot_arbitrage_horizon_cone(horizon_data_3d, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, horizon_data_2d) -> None:
        fig = plot_arbitrage_horizon_cone(
            horizon_data_2d, backend="plotly", title="Cone"
        )
        assert fig.layout.title.text == "Cone"


class TestArbitrageHorizonConeMatplotlib:
    def test_returns_mpl_figure(self, horizon_data_2d) -> None:
        fig = plot_arbitrage_horizon_cone(horizon_data_2d, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, horizon_data_2d) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_arbitrage_horizon_cone(
            horizon_data_2d, backend="matplotlib", ax=ax
        )
        assert isinstance(result, MplFigure)
        plt.close(fig)
