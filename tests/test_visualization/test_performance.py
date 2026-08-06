"""Tests for visualization/performance.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.performance import (
    plot_cumulative_returns,
    plot_master_formula_decomposition,
    plot_relative_performance,
)


@pytest.fixture()
def returns_dict():
    rng = np.random.default_rng(42)
    return {
        "Portfolio": rng.normal(0.001, 0.02, size=100),
        "Market": rng.normal(0.0005, 0.015, size=100),
    }


@pytest.fixture()
def pi_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, size=100)


@pytest.fixture()
def mu_returns():
    rng = np.random.default_rng(43)
    return rng.normal(0.0005, 0.015, size=100)


@pytest.fixture()
def decomp_dict():
    rng = np.random.default_rng(42)
    t = 100
    boundary = np.cumsum(rng.normal(0.0001, 0.001, size=t))
    drift = np.cumsum(rng.normal(-0.00005, 0.0005, size=t))
    return {"boundary": boundary, "drift": drift}


class TestCumulativeReturnsPlotly:
    def test_returns_plotly_figure(self, returns_dict) -> None:
        fig = plot_cumulative_returns(returns_dict, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, returns_dict) -> None:
        fig = plot_cumulative_returns(returns_dict, backend="plotly", title="Test")
        assert fig.layout.title.text == "Test"

    def test_single_strategy(self) -> None:
        rng = np.random.default_rng(42)
        fig = plot_cumulative_returns(
            {"Solo": rng.normal(0, 0.01, 50)}, backend="plotly"
        )
        assert isinstance(fig, go.Figure)


class TestCumulativeReturnsMatplotlib:
    def test_returns_mpl_figure(self, returns_dict) -> None:
        fig = plot_cumulative_returns(returns_dict, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, returns_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_cumulative_returns(returns_dict, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)


class TestRelativePerformancePlotly:
    def test_returns_plotly_figure(self, pi_returns, mu_returns) -> None:
        fig = plot_relative_performance(pi_returns, mu_returns, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_identical_returns(self) -> None:
        rets = np.zeros(50)
        fig = plot_relative_performance(rets, rets, backend="plotly")
        assert isinstance(fig, go.Figure)


class TestRelativePerformanceMatplotlib:
    def test_returns_mpl_figure(self, pi_returns, mu_returns) -> None:
        fig = plot_relative_performance(pi_returns, mu_returns, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestMasterFormulaPlotly:
    def test_returns_plotly_figure(self, decomp_dict) -> None:
        fig = plot_master_formula_decomposition(decomp_dict, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_with_residual(self, decomp_dict) -> None:
        decomp_dict["residual"] = np.zeros(100)
        fig = plot_master_formula_decomposition(decomp_dict, backend="plotly")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4


class TestMasterFormulaMatplotlib:
    def test_returns_mpl_figure(self, decomp_dict) -> None:
        fig = plot_master_formula_decomposition(decomp_dict, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, decomp_dict) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_master_formula_decomposition(
            decomp_dict, backend="matplotlib", ax=ax
        )
        assert isinstance(result, MplFigure)
        plt.close(fig)
