"""Tests for visualization/model_diagnostics.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.model_diagnostics import (
    plot_convergence,
    plot_qq,
    plot_residuals,
)


@pytest.fixture()
def normal_residuals():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


@pytest.fixture()
def loss_history():
    return np.exp(-np.linspace(0, 3, 100))


class TestQQPlotly:
    def test_returns_plotly_figure(self, normal_residuals) -> None:
        fig = plot_qq(normal_residuals, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, normal_residuals) -> None:
        fig = plot_qq(normal_residuals, backend="plotly", title="Custom QQ")
        assert fig.layout.title.text == "Custom QQ"

    def test_small_sample(self) -> None:
        fig = plot_qq(np.array([1.0, 2.0]), backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_large_sample(self) -> None:
        rng = np.random.default_rng(99)
        fig = plot_qq(rng.standard_normal(1000), backend="plotly")
        assert isinstance(fig, go.Figure)


class TestQQMatplotlib:
    def test_returns_mpl_figure(self, normal_residuals) -> None:
        fig = plot_qq(normal_residuals, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, normal_residuals) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_qq(normal_residuals, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestConvergencePlotly:
    def test_returns_plotly_figure(self, loss_history) -> None:
        fig = plot_convergence(loss_history, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_log_scale(self, loss_history) -> None:
        fig = plot_convergence(loss_history, backend="plotly", log_scale=True)
        assert isinstance(fig, go.Figure)

    def test_single_iteration(self) -> None:
        fig = plot_convergence(np.array([0.5]), backend="plotly")
        assert isinstance(fig, go.Figure)


class TestConvergenceMatplotlib:
    def test_returns_mpl_figure(self, loss_history) -> None:
        fig = plot_convergence(loss_history, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, loss_history) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_convergence(loss_history, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestResidualsPlotly:
    def test_returns_plotly_figure(self, normal_residuals) -> None:
        fig = plot_residuals(normal_residuals, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, normal_residuals) -> None:
        fig = plot_residuals(normal_residuals, backend="plotly", title="Resid")
        assert fig.layout.title.text == "Resid"

    def test_empty_edge_case(self) -> None:
        fig = plot_residuals(np.array([0.0]), backend="plotly")
        assert isinstance(fig, go.Figure)


class TestResidualsMatplotlib:
    def test_returns_mpl_figure(self, normal_residuals) -> None:
        fig = plot_residuals(normal_residuals, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, normal_residuals) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_residuals(normal_residuals, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)
