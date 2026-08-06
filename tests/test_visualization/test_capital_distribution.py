"""Tests for visualization/capital_distribution.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.capital_distribution import (
    plot_capital_distribution,
    plot_capital_distribution_evolution,
)


@pytest.fixture()
def sample_weights():
    rng = np.random.default_rng(42)
    return rng.dirichlet(np.ones(10))


@pytest.fixture()
def weight_paths():
    rng = np.random.default_rng(42)
    paths = np.zeros((50, 5))
    w = rng.dirichlet(np.ones(5))
    for t in range(50):
        noise = rng.standard_normal(5) * 0.01
        w = np.abs(w + noise)
        w /= w.sum()
        paths[t] = w
    return paths


class TestCapitalDistributionPlotly:
    def test_returns_plotly_figure(self, sample_weights) -> None:
        fig = plot_capital_distribution(sample_weights, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_log_scale(self, sample_weights) -> None:
        fig = plot_capital_distribution(
            sample_weights, backend="plotly", log_scale=True
        )
        assert isinstance(fig, go.Figure)

    def test_linear_scale(self, sample_weights) -> None:
        fig = plot_capital_distribution(
            sample_weights, backend="plotly", log_scale=False
        )
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, sample_weights) -> None:
        fig = plot_capital_distribution(
            sample_weights, backend="plotly", title="Custom"
        )
        assert fig.layout.title.text == "Custom"

    def test_with_labels(self, sample_weights) -> None:
        labels = [f"Stock_{i}" for i in range(10)]
        fig = plot_capital_distribution(sample_weights, backend="plotly", labels=labels)
        assert isinstance(fig, go.Figure)

    def test_single_stock(self) -> None:
        fig = plot_capital_distribution(np.array([1.0]), backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_100_stocks(self) -> None:
        rng = np.random.default_rng(99)
        w = rng.dirichlet(np.ones(100))
        fig = plot_capital_distribution(w, backend="plotly")
        assert isinstance(fig, go.Figure)


class TestCapitalDistributionMatplotlib:
    def test_returns_mpl_figure(self, sample_weights) -> None:
        fig = plot_capital_distribution(sample_weights, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, sample_weights) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_capital_distribution(sample_weights, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)

    def test_linear_scale(self, sample_weights) -> None:
        fig = plot_capital_distribution(
            sample_weights, backend="matplotlib", log_scale=False
        )
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestCapitalDistributionEvolutionPlotly:
    def test_returns_plotly_figure(self, weight_paths) -> None:
        fig = plot_capital_distribution_evolution(weight_paths, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_with_times(self, weight_paths) -> None:
        times = np.linspace(0, 1, 50)
        fig = plot_capital_distribution_evolution(
            weight_paths, backend="plotly", times=times
        )
        assert isinstance(fig, go.Figure)

    def test_n_snapshots(self, weight_paths) -> None:
        fig = plot_capital_distribution_evolution(
            weight_paths, backend="plotly", n_snapshots=3
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3


class TestCapitalDistributionEvolutionMatplotlib:
    def test_returns_mpl_figure(self, weight_paths) -> None:
        fig = plot_capital_distribution_evolution(weight_paths, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weight_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_capital_distribution_evolution(
            weight_paths, backend="matplotlib", ax=ax
        )
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestInvalidBackend:
    def test_raises_on_invalid(self, sample_weights) -> None:
        with pytest.raises(ValueError, match="backend"):
            plot_capital_distribution(sample_weights, backend="invalid")  # type: ignore[arg-type]
