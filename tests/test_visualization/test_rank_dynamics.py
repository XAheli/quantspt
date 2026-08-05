"""Tests for visualization/rank_dynamics.py."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

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
    mat = rng.dirichlet(np.ones(n), size=n)
    return mat


class TestRankChanges:
    def test_returns_figure(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, rank_paths) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_rank_changes(rank_paths, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_n_assets_limit(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, n_assets=3)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_title(self, rank_paths) -> None:
        fig = plot_rank_changes(rank_paths, title="Rank Test")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRankTransitionHeatmap:
    def test_returns_figure(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(transition_matrix)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_ax(self, transition_matrix) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_rank_transition_heatmap(transition_matrix, ax=ax)
        assert isinstance(result, Figure)
        plt.close(fig)

    def test_custom_cmap(self, transition_matrix) -> None:
        fig = plot_rank_transition_heatmap(transition_matrix, cmap="Reds")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_identity_matrix(self) -> None:
        mat = np.eye(4)
        fig = plot_rank_transition_heatmap(mat)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)
