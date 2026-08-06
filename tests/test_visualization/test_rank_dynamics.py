"""Tests for visualization/rank_dynamics.py — dual backend."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from quantspt.visualization.rank_dynamics import (
    plot_rank_changes,
    plot_rank_transition_heatmap,
)


@pytest.fixture()
def rank_paths():
    rng = np.random.default_rng(42)
    n_time, n_assets = 50, 5
    ranks = np.zeros((n_time, n_assets), dtype=np.intp)
    for t in range(n_time):
        ranks[t] = rng.permutation(n_assets)
    return ranks


@pytest.fixture()
def transition_matrix():
    n = 5
    rng = np.random.default_rng(42)
    return rng.dirichlet(np.ones(n), size=n)


class TestRankChangesPlotly:
    def test_returns_plotly_figure(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_n_assets_limit(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, backend="plotly", n_assets=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_with_labels(self, rank_paths) -> None:
        labels = [f"Stock {i}" for i in range(5)]
        fig = plot_rank_changes(rank_paths, backend="plotly", labels=labels)
        assert isinstance(fig, go.Figure)

    def test_single_asset(self) -> None:
        ranks = np.zeros((20, 1), dtype=np.intp)
        fig = plot_rank_changes(ranks, backend="plotly")
        assert isinstance(fig, go.Figure)


class TestRankChangesMatplotlib:
    def test_returns_mpl_figure(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, rank_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_rank_changes(rank_paths, backend="matplotlib", ax=ax)
        assert isinstance(result, MplFigure)
        plt.close(fig)

    def test_n_assets_limit(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, backend="matplotlib", n_assets=3)
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRankTransitionHeatmapPlotly:
    def test_returns_plotly_figure(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(transition_matrix, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_identity_matrix(self) -> None:
        fig = plot_rank_transition_heatmap(np.eye(4), backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(
            transition_matrix, backend="plotly", title="Test"
        )
        assert fig.layout.title.text == "Test"


class TestRankTransitionHeatmapMatplotlib:
    def test_returns_mpl_figure(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(transition_matrix, backend="matplotlib")
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, transition_matrix) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_rank_transition_heatmap(
            transition_matrix, backend="matplotlib", ax=ax
        )
        assert isinstance(result, MplFigure)
        plt.close(fig)

    def test_custom_cmap(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(
            transition_matrix, backend="matplotlib", cmap="Reds"
        )
        assert isinstance(fig, MplFigure)
        import matplotlib.pyplot as plt

        plt.close(fig)
