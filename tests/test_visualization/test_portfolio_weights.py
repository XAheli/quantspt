"""Tests for visualization/portfolio_weights.py."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from quantspt.visualization.portfolio_weights import (
    plot_weight_comparison,
    plot_weight_evolution,
)


@pytest.fixture()
def weight_paths():
    rng = np.random.default_rng(42)
    paths = np.zeros((30, 4))
    w = rng.dirichlet(np.ones(4))
    for t in range(30):
        noise = rng.standard_normal(4) * 0.01
        w = np.abs(w + noise)
        w /= w.sum()
        paths[t] = w
    return paths


@pytest.fixture()
def weights_dict():
    rng = np.random.default_rng(42)
    return {
        "Equal weight": np.ones(5) / 5,
        "Diversity": rng.dirichlet(np.ones(5)),
        "Concentrated": np.array([0.5, 0.2, 0.15, 0.1, 0.05]),
    }


class TestWeightEvolution:
    def test_returns_figure(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_stacked(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, stacked=True)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_labels(self, weight_paths) -> None:
        labels = [f"Stock {i}" for i in range(4)]
        fig = plot_weight_evolution(weight_paths, labels=labels)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weight_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_weight_evolution(weight_paths, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_custom_title(self, weight_paths) -> None:
        fig = plot_weight_evolution(weight_paths, title="Weights")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestWeightComparison:
    def test_returns_figure(self, weights_dict) -> None:
        fig = plot_weight_comparison(weights_dict)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_tickers(self, weights_dict) -> None:
        tickers = ["A", "B", "C", "D", "E"]
        fig = plot_weight_comparison(weights_dict, tickers=tickers)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weights_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_weight_comparison(weights_dict, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_single_strategy(self) -> None:
        d = {"Only": np.array([0.5, 0.3, 0.2])}
        fig = plot_weight_comparison(d)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)
