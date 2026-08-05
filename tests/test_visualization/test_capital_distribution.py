"""Tests for visualization/capital_distribution.py."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from quantspt.visualization.capital_distribution import (
    plot_capital_distribution,
    plot_capital_distribution_evolution,
)


@pytest.fixture()
def sample_weights():
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(10))
    return w


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


class TestCapitalDistribution:
    def test_returns_figure(self, sample_weights) -> None:
        fig = plot_capital_distribution(sample_weights)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, sample_weights) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_capital_distribution(sample_weights, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_linear_scale(self, sample_weights) -> None:
        fig = plot_capital_distribution(sample_weights, log_scale=False)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_title(self, sample_weights) -> None:
        fig = plot_capital_distribution(sample_weights, title="Test Title")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestCapitalDistributionEvolution:
    def test_returns_figure(self, weight_paths) -> None:
        fig = plot_capital_distribution_evolution(weight_paths)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_times(self, weight_paths) -> None:
        times = np.linspace(0, 1, 50)
        fig = plot_capital_distribution_evolution(weight_paths, times=times)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_n_snapshots(self, weight_paths) -> None:
        fig = plot_capital_distribution_evolution(weight_paths, n_snapshots=3)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, weight_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_capital_distribution_evolution(weight_paths, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)
