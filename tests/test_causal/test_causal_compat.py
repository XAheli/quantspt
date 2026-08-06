"""Comprehensive pgmpy/causal compatibility tests.

Validates that:
- CausalStructureLearner with method="pc" recovers a known 3-node chain
- CausalStructureLearner with method="ges" recovers same DAG
- CausalStructureLearner with method="hillclimb" recovers same DAG
- Different scoring methods: "bic-g", "aic-g" both work
- Different CI tests: "pearsonr" works on continuous data
- User passes own LinearGaussianBayesianNetwork → works
- observational_covariance matches numpy cov (up to finite-sample)
- interventional_covariance actually changes the matrix vs observational
- GPU (torch backend): results match CPU (numpy backend)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.causal.covariance import CausalCovarianceEstimator
from quantspt.causal.structure import CausalStructureLearner

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def chain_data_large(rng) -> pd.DataFrame:
    """Large synthetic chain X → Y → Z for reliable recovery."""
    n = 2000
    X = rng.normal(size=n)
    Y = 0.9 * X + rng.normal(size=n) * 0.2
    Z = 0.7 * Y + rng.normal(size=n) * 0.15
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def fork_data_large(rng) -> pd.DataFrame:
    """Large fork X → Y, X → Z."""
    n = 2000
    X = rng.normal(size=n)
    Y = 0.8 * X + rng.normal(size=n) * 0.3
    Z = 0.7 * X + rng.normal(size=n) * 0.2
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


# ---------------------------------------------------------------------------
# DAG recovery across algorithms
# ---------------------------------------------------------------------------


class TestDAGRecoveryAllMethods:
    """All methods recover the known chain DAG."""

    @pytest.mark.parametrize("method", ["pc", "ges", "hillclimb"])
    def test_chain_recovery(self, method, chain_data_large) -> None:
        """Method recovers X-Y and Y-Z edges from chain data."""
        kwargs = {}
        if method in ("ges", "hillclimb"):
            kwargs["scoring_method"] = "bic-g"
        if method == "pc":
            kwargs["ci_test"] = "pearsonr"
            kwargs["significance_level"] = 0.01

        learner = CausalStructureLearner(method=method, **kwargs)
        learner.fit(chain_data_large)
        edge_set = {frozenset(e) for e in learner.edges}
        assert frozenset(("X", "Y")) in edge_set
        assert frozenset(("Y", "Z")) in edge_set

    @pytest.mark.parametrize("method", ["pc", "ges", "hillclimb"])
    def test_fork_recovery(self, method, fork_data_large) -> None:
        """Method recovers X-Y and X-Z edges from fork data."""
        kwargs = {}
        if method in ("ges", "hillclimb"):
            kwargs["scoring_method"] = "bic-g"
        if method == "pc":
            kwargs["ci_test"] = "pearsonr"
            kwargs["significance_level"] = 0.01

        learner = CausalStructureLearner(method=method, **kwargs)
        learner.fit(fork_data_large)
        edge_set = {frozenset(e) for e in learner.edges}
        assert frozenset(("X", "Y")) in edge_set
        assert frozenset(("X", "Z")) in edge_set


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------


class TestScoringMethods:
    """Different scoring methods work on continuous data."""

    @pytest.mark.parametrize("score", ["bic-g", "aic-g"])
    def test_ges_scoring(self, score, chain_data_large) -> None:
        learner = CausalStructureLearner(method="ges", scoring_method=score)
        learner.fit(chain_data_large)
        assert len(learner.edges) > 0

    @pytest.mark.parametrize("score", ["bic-g", "aic-g"])
    def test_hillclimb_scoring(self, score, chain_data_large) -> None:
        learner = CausalStructureLearner(method="hillclimb", scoring_method=score)
        learner.fit(chain_data_large)
        assert len(learner.edges) > 0


# ---------------------------------------------------------------------------
# CI tests
# ---------------------------------------------------------------------------


class TestCITests:
    """pearsonr CI test works on continuous data."""

    def test_pearsonr_finds_edges(self, chain_data_large) -> None:
        learner = CausalStructureLearner(
            method="pc", ci_test="pearsonr", significance_level=0.01
        )
        learner.fit(chain_data_large)
        assert len(learner.edges) >= 2


# ---------------------------------------------------------------------------
# User-provided LinearGaussianBayesianNetwork
# ---------------------------------------------------------------------------


class TestUserProvidedModel:
    """User passes own fitted LinearGaussianBayesianNetwork."""

    def test_from_pgmpy_model_works(self, chain_data_large) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(chain_data_large)

        learner = CausalStructureLearner.from_pgmpy(model)
        assert learner.edges == [("X", "Y"), ("Y", "Z")]
        assert learner.adjacency_matrix.shape == (3, 3)
        assert set(learner.variable_names) == {"X", "Y", "Z"}

    def test_from_pgmpy_adjacency_correct(self, chain_data_large) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(chain_data_large)

        learner = CausalStructureLearner.from_pgmpy(model)
        adj = learner.adjacency_matrix
        names = learner.variable_names
        idx = {n: i for i, n in enumerate(names)}

        assert adj[idx["X"], idx["Y"]] == 1.0
        assert adj[idx["Y"], idx["Z"]] == 1.0
        assert adj[idx["X"], idx["Z"]] == 0.0


# ---------------------------------------------------------------------------
# Observational covariance matches numpy cov
# ---------------------------------------------------------------------------


class TestObservationalCovariance:
    """observational_covariance matches numpy cov (up to finite-sample)."""

    def test_obs_cov_close_to_sample_cov(self, rng) -> None:
        """Model covariance approximates sample covariance for large n."""
        n_samples = 5000
        X = rng.normal(size=n_samples)
        Y = 0.8 * X + rng.normal(size=n_samples) * 0.3
        Z = 0.5 * Y + rng.normal(size=n_samples) * 0.2
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(data)
        model_cov = estimator.observational_covariance()

        sample_cov = np.cov(data.values, rowvar=False)
        names = estimator.variable_names
        col_order = [data.columns.get_loc(n) for n in names]
        sample_cov_reordered = sample_cov[np.ix_(col_order, col_order)]

        assert_allclose(model_cov, sample_cov_reordered, atol=0.05, rtol=0.1)

    def test_obs_cov_is_symmetric(self, chain_data_large) -> None:
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)
        cov = estimator.observational_covariance()
        assert_allclose(cov, cov.T, atol=1e-10)

    def test_obs_cov_is_psd(self, chain_data_large) -> None:
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)
        cov = estimator.observational_covariance()
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)


# ---------------------------------------------------------------------------
# Interventional covariance differs from observational
# ---------------------------------------------------------------------------


class TestInterventionalCovariance:
    """interventional_covariance actually changes the matrix."""

    def test_intervention_changes_matrix(self, chain_data_large) -> None:
        """do(X=0) must change the covariance relative to observational."""
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)

        obs_cov = estimator.observational_covariance()
        int_cov = estimator.interventional_covariance({"X": 0.0})

        assert not np.allclose(obs_cov, int_cov, atol=1e-3)

    def test_intervention_zeros_intervened_variance(self, chain_data_large) -> None:
        """Intervened variable has zero variance."""
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)

        int_cov = estimator.interventional_covariance({"X": 0.0})
        names = estimator.variable_names
        x_idx = names.index("X")
        assert_allclose(int_cov[x_idx, :], 0.0, atol=1e-10)
        assert_allclose(int_cov[:, x_idx], 0.0, atol=1e-10)

    def test_downstream_variance_reduced(self, chain_data_large) -> None:
        """Intervention on X should reduce variance of downstream Y."""
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)

        obs_cov = estimator.observational_covariance()
        int_cov = estimator.interventional_covariance({"X": 0.0})
        names = estimator.variable_names
        y_idx = names.index("Y")

        assert int_cov[y_idx, y_idx] < obs_cov[y_idx, y_idx]


# ---------------------------------------------------------------------------
# Decomposition consistency
# ---------------------------------------------------------------------------


class TestCovarianceDecomposition:
    """Structural decomposition Σ = (I-B)⁻¹ Ω (I-B)⁻ᵀ is correct."""

    def test_decomposition_matches_obs_cov(self, chain_data_large) -> None:
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)

        decomp = estimator.decompose()
        obs_cov = estimator.observational_covariance()
        assert_allclose(decomp.sigma, obs_cov, atol=0.05, rtol=0.1)

    def test_B_matrix_encodes_edges(self, chain_data_large) -> None:
        estimator = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        estimator.fit(chain_data_large)

        decomp = estimator.decompose()
        names = decomp.variable_names
        idx = {n: i for i, n in enumerate(names)}

        assert abs(decomp.B[idx["Y"], idx["X"]]) > 0.5
        assert abs(decomp.B[idx["Z"], idx["Y"]]) > 0.3
        assert abs(decomp.B[idx["Z"], idx["X"]]) < 0.01


# ---------------------------------------------------------------------------
# GPU/CPU consistency (pgmpy torch backend)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCausalStructureGPUConsistency:
    """Verify pgmpy torch backend on CUDA matches numpy backend results."""

    def test_ges_adjacency_cpu_gpu_match(self, chain_data_large) -> None:
        """GES discovery produces same adjacency on CPU and GPU."""
        learner_cpu = CausalStructureLearner(
            method="ges", scoring_method="bic-g", backend="numpy"
        )
        learner_cpu.fit(chain_data_large)
        adj_cpu = learner_cpu.adjacency_matrix

        learner_gpu = CausalStructureLearner(
            method="ges", scoring_method="bic-g", backend="torch", device="cuda"
        )
        learner_gpu.fit(chain_data_large)
        adj_gpu = learner_gpu.adjacency_matrix

        assert_allclose(adj_cpu, adj_gpu)

    def test_hillclimb_adjacency_cpu_gpu_match(self, chain_data_large) -> None:
        """HillClimb produces same adjacency on CPU and GPU."""
        learner_cpu = CausalStructureLearner(
            method="hillclimb", scoring_method="bic-g", backend="numpy"
        )
        learner_cpu.fit(chain_data_large)
        adj_cpu = learner_cpu.adjacency_matrix

        learner_gpu = CausalStructureLearner(
            method="hillclimb", scoring_method="bic-g", backend="torch", device="cuda"
        )
        learner_gpu.fit(chain_data_large)
        adj_gpu = learner_gpu.adjacency_matrix

        assert_allclose(adj_cpu, adj_gpu)

    def test_pc_edges_cpu_gpu_match(self, chain_data_large) -> None:
        """PC algorithm finds same edges on CPU and GPU."""
        learner_cpu = CausalStructureLearner(
            method="pc",
            ci_test="pearsonr",
            significance_level=0.01,
            backend="numpy",
        )
        learner_cpu.fit(chain_data_large)
        edges_cpu = {frozenset(e) for e in learner_cpu.edges}

        learner_gpu = CausalStructureLearner(
            method="pc",
            ci_test="pearsonr",
            significance_level=0.01,
            backend="torch",
            device="cuda",
        )
        learner_gpu.fit(chain_data_large)
        edges_gpu = {frozenset(e) for e in learner_gpu.edges}

        assert edges_cpu == edges_gpu


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCausalCovarianceGPUConsistency:
    """Verify CausalCovarianceEstimator GPU matches CPU."""

    def test_observational_cov_cpu_gpu_match(self, chain_data_large) -> None:
        """Observational covariance matches between numpy and torch/CUDA."""
        est_cpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="numpy"
        )
        est_cpu.fit(chain_data_large)
        cov_cpu = est_cpu.observational_covariance()

        est_gpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="torch", device="cuda"
        )
        est_gpu.fit(chain_data_large)
        cov_gpu = est_gpu.observational_covariance()

        assert_allclose(cov_cpu, cov_gpu, atol=1e-6)

    def test_interventional_cov_cpu_gpu_structure_match(self, chain_data_large) -> None:
        """Interventional covariance has same zero-pattern on CPU and GPU.

        The exact values may differ slightly because interventional covariance
        relies on internal simulation with different random states across backends.
        We verify structural properties are preserved.
        """
        est_cpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="numpy"
        )
        est_cpu.fit(chain_data_large)
        int_cpu = est_cpu.interventional_covariance({"X": 0.0})

        est_gpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="torch", device="cuda"
        )
        est_gpu.fit(chain_data_large)
        int_gpu = est_gpu.interventional_covariance({"X": 0.0})

        names = est_cpu.variable_names
        x_idx = names.index("X")
        assert_allclose(int_cpu[x_idx, :], 0.0, atol=1e-10)
        assert_allclose(int_gpu[x_idx, :], 0.0, atol=1e-10)
        assert_allclose(int_cpu[:, x_idx], 0.0, atol=1e-10)
        assert_allclose(int_gpu[:, x_idx], 0.0, atol=1e-10)

        assert int_gpu.shape == int_cpu.shape
        assert_allclose(int_gpu, int_gpu.T, atol=1e-10)

    def test_decomposition_cpu_gpu_match(self, chain_data_large) -> None:
        """Structural decomposition matches between backends."""
        est_cpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="numpy"
        )
        est_cpu.fit(chain_data_large)
        decomp_cpu = est_cpu.decompose()

        est_gpu = CausalCovarianceEstimator(
            edges=[("X", "Y"), ("Y", "Z")], backend="torch", device="cuda"
        )
        est_gpu.fit(chain_data_large)
        decomp_gpu = est_gpu.decompose()

        assert_allclose(decomp_cpu.B, decomp_gpu.B, atol=1e-6)
        assert_allclose(decomp_cpu.omega, decomp_gpu.omega, atol=1e-6)
        assert_allclose(decomp_cpu.sigma, decomp_gpu.sigma, atol=1e-6)
