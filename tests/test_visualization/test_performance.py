"""Tests for visualization/performance.py."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from quantspt.visualization.performance import (
    plot_cumulative_returns,
    plot_master_formula_decomposition,
    plot_relative_performance,
)


@pytest.fixture()
def returns_dict():
    rng = np.random.default_rng(42)
    return {
        "Strategy A": rng.standard_normal(100) * 0.01,
        "Strategy B": rng.standard_normal(100) * 0.01 + 0.001,
    }


@pytest.fixture()
def decomp_dict():
    t = np.linspace(0, 1, 50)
    return {
        "boundary": -0.003 * t,
        "drift": 0.014 * t,
        "residual": np.random.default_rng(42).standard_normal(50) * 0.001,
    }


class TestCumulativeReturns:
    def test_returns_figure(self, returns_dict) -> None:
        fig = plot_cumulative_returns(returns_dict)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, returns_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_cumulative_returns(returns_dict, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_custom_title(self, returns_dict) -> None:
        fig = plot_cumulative_returns(returns_dict, title="My Title")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRelativePerformance:
    def test_returns_figure(self) -> None:
        rng = np.random.default_rng(42)
        pi = rng.standard_normal(50) * 0.01
        mu = rng.standard_normal(50) * 0.01
        fig = plot_relative_performance(pi, mu)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self) -> None:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        pi = rng.standard_normal(50) * 0.01
        mu = rng.standard_normal(50) * 0.01
        fig, ax = plt.subplots()
        result = plot_relative_performance(pi, mu, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)


class TestMasterFormulaDecomp:
    def test_returns_figure(self, decomp_dict) -> None:
        fig = plot_master_formula_decomposition(decomp_dict)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_residual(self) -> None:
        t = np.linspace(0, 1, 20)
        d = {"boundary": -0.01 * t, "drift": 0.02 * t}
        fig = plot_master_formula_decomposition(d)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, decomp_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_master_formula_decomposition(decomp_dict, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)
