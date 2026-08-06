"""Tests for visualization/portfolio_weights.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.portfolio_weights import (
    plot_weight_comparison,
    plot_weight_evolution,
)


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


@pytest.fixture()
def weights_dict():
    rng = np.random.default_rng(42)
    return {
        "Strategy A": rng.dirichlet(np.ones(5)),
        "Strategy B": rng.dirichlet(np.ones(5)),
    }


class TestWeightEvolutionPlotly:
    def test_returns_plotly_figure(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_stacked(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, backend="plotly", stacked=True)
        assert isinstance(fig, go.Figure)

    def test_with_labels(self, weight_paths) -> None:
        labels = [f"Asset {i}" for i in range(5)]
        fig = plot_weight_evolution(weight_paths, backend="plotly", labels=labels)
        assert isinstance(fig, go.Figure)

    def test_single_asset(self) -> None:
        paths = np.ones((20, 1))
        fig = plot_weight_evolution(paths, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_100_assets(self) -> None:
        rng = np.random.default_rng(99)
        paths = np.zeros((10, 100))
        for t in range(10):
            paths[t] = rng.dirichlet(np.ones(100))
        fig = plot_weight_evolution(paths, backend="plotly")
        assert isinstance(fig, go.Figure)


class TestWeightEvolutionMatplotlib:
    def test_returns_mpl_figure(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_stacked(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, backend="matplotlib", stacked=True)
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weight_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_weight_evolution(weight_paths, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestWeightComparisonPlotly:
    def test_returns_plotly_figure(self, weights_dict) -> None:
        fig = plot_weight_comparison(weights_dict, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_with_tickers(self, weights_dict) -> None:
        tickers = ["A", "B", "C", "D", "E"]
        fig = plot_weight_comparison(weights_dict, backend="plotly", tickers=tickers)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, weights_dict) -> None:
        fig = plot_weight_comparison(weights_dict, backend="plotly", title="Custom")
        assert fig.layout.title.text == "Custom"


class TestWeightComparisonMatplotlib:
    def test_returns_mpl_figure(self, weights_dict) -> None:
        fig = plot_weight_comparison(weights_dict, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weights_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_weight_comparison(weights_dict, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)
