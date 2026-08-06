"""Tests for causal covariance estimation and do-calculus.

Validates:
- Observational covariance matches numpy computation on known parameters
- Structural decomposition Σ = (I-B)^{-1} Ω (I-B)^{-T}
- Interventional covariance removes correct edges
- Pluggability: user provides their own LGBN
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.causal.covariance import (
    CausalCovarianceEstimator,
    CovarianceDecomposition,
)
from quantspt.errors import SPTInvariantError


@pytest.fixture()
def known_sem_data() -> tuple[pd.DataFrame, dict[str, float]]:
    """Synthetic data from a known linear SEM.

    True parameters:
        X ~ N(0, 1)
        Y = 0.8*X + ε_Y,  ε_Y ~ N(0, 0.09)
        Z = 0.5*Y + ε_Z,  ε_Z ~ N(0, 0.04)
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(size=n)
    Y = 0.8 * X + rng.normal(scale=0.3, size=n)
    Z = 0.5 * Y + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    true_params = {
        "b_yx": 0.8,
        "b_zy": 0.5,
        "var_x": 1.0,
        "var_ey": 0.09,
        "var_ez": 0.04,
    }
    return df, true_params


@pytest.fixture()
def three_node_data() -> pd.DataFrame:
    """Data with X -> Y -> Z and X -> Z (triangle)."""
    rng = np.random.default_rng(123)
    n = 1000
    X = rng.normal(size=n)
    Y = 0.7 * X + rng.normal(scale=0.4, size=n)
    Z = 0.5 * Y + 0.3 * X + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


class TestObservationalCovariance:
    """Observational covariance Σ via to_joint_gaussian()."""

    def test_shape_and_symmetry(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        cov = est.observational_covariance()
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_positive_semidefinite(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        cov = est.observational_covariance()
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_matches_decomposition(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        """Σ from to_joint_gaussian() must match (I-B)^{-1} Ω (I-B)^{-T}."""
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)

        obs_cov = est.observational_covariance()
        decomp = est.decompose()
        np.testing.assert_allclose(obs_cov, decomp.sigma, atol=1e-8)

    def test_diagonal_positive(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        cov = est.observational_covariance()
        assert np.all(np.diag(cov) > 0)


class TestDecomposition:
    """Structural decomposition Σ = (I-B)^{-1} Ω (I-B)^{-T}."""

    def test_b_matrix_recovers_true_coefficients(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, true_params = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        decomp = est.decompose()

        names = decomp.variable_names
        idx = {n: i for i, n in enumerate(names)}

        np.testing.assert_allclose(
            decomp.B[idx["Y"], idx["X"]], true_params["b_yx"], atol=0.05
        )
        np.testing.assert_allclose(
            decomp.B[idx["Z"], idx["Y"]], true_params["b_zy"], atol=0.05
        )
        assert decomp.B[idx["X"], idx["Y"]] == 0.0
        assert decomp.B[idx["X"], idx["Z"]] == 0.0

    def test_omega_recovers_noise_variances(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, true_params = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        decomp = est.decompose()

        idx = {n: i for i, n in enumerate(decomp.variable_names)}

        np.testing.assert_allclose(
            decomp.omega[idx["X"]], true_params["var_x"], atol=0.1
        )
        np.testing.assert_allclose(
            decomp.omega[idx["Y"]], true_params["var_ey"], atol=0.03
        )
        np.testing.assert_allclose(
            decomp.omega[idx["Z"]], true_params["var_ez"], atol=0.02
        )

    def test_decomposition_formula_identity(
        self,
        three_node_data: pd.DataFrame,
    ) -> None:
        """Verify Σ = (I-B)^{-1} Ω (I-B)^{-T} directly."""
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z"), ("X", "Z")])
        est.fit(three_node_data)
        decomp = est.decompose()

        n = len(decomp.variable_names)
        I_minus_B_inv = np.linalg.inv(np.eye(n) - decomp.B)
        expected = I_minus_B_inv @ np.diag(decomp.omega) @ I_minus_B_inv.T
        np.testing.assert_allclose(decomp.sigma, expected, atol=1e-10)

    def test_decomposition_dataclass_fields(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        decomp = est.decompose()

        assert isinstance(decomp, CovarianceDecomposition)
        assert decomp.B.shape == (3, 3)
        assert decomp.omega.shape == (3,)
        assert decomp.sigma.shape == (3, 3)
        assert len(decomp.variable_names) == 3


class TestInterventionalCovariance:
    """Interventional covariance via do-calculus (graph mutilation)."""

    def test_intervention_zeroes_intervened_variable(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)

        int_cov = est.interventional_covariance({"X": 1.0})

        idx = {n: i for i, n in enumerate(est.variable_names)}
        x_idx = idx["X"]

        assert int_cov[x_idx, x_idx] == 0.0
        np.testing.assert_allclose(int_cov[x_idx, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(int_cov[:, x_idx], 0.0, atol=1e-12)

    def test_intervention_preserves_downstream(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        """do(X=c) should not affect the Y→Z relationship."""
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)

        int_cov = est.interventional_covariance({"X": 1.0})

        idx = {n: i for i, n in enumerate(est.variable_names)}
        assert int_cov[idx["Y"], idx["Y"]] > 0
        assert int_cov[idx["Z"], idx["Z"]] > 0

    def test_intervention_shape(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        int_cov = est.interventional_covariance({"Y": 0.0})
        assert int_cov.shape == (3, 3)

    def test_intervention_reduces_variance(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        """Intervening on X should reduce downstream variance."""
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)

        obs_cov = est.observational_covariance()
        int_cov = est.interventional_covariance({"X": 0.0})

        idx = {n: i for i, n in enumerate(est.variable_names)}
        assert int_cov[idx["Y"], idx["Y"]] < obs_cov[idx["Y"], idx["Y"]]


class TestFromPgmpy:
    """Pluggability: user provides their own LGBN."""

    def test_from_pgmpy_observational(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        df, _ = known_sem_data
        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(df)

        est = CausalCovarianceEstimator.from_pgmpy(model)
        cov = est.observational_covariance()
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_from_pgmpy_decomposition(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        df, _ = known_sem_data
        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(df)

        est = CausalCovarianceEstimator.from_pgmpy(model)
        decomp = est.decompose()
        obs_cov = est.observational_covariance()
        np.testing.assert_allclose(decomp.sigma, obs_cov, atol=1e-8)


class TestEdgeConfiguration:
    """Test edge configuration at init vs fit time."""

    def test_edges_at_init(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        assert est.observational_covariance().shape == (3, 3)

    def test_edges_at_fit(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator()
        est.fit(df, edges=[("X", "Y"), ("Y", "Z")])
        assert est.observational_covariance().shape == (3, 3)

    def test_no_edges_raises(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator()
        with pytest.raises(SPTInvariantError, match="edges"):
            est.fit(df)


class TestValidation:
    """Input validation and access-before-fit guards."""

    def test_access_before_fit_raises(self) -> None:
        est = CausalCovarianceEstimator(edges=[("X", "Y")])
        with pytest.raises(SPTInvariantError, match="fit"):
            est.observational_covariance()
        with pytest.raises(SPTInvariantError, match="fit"):
            est.decompose()
        with pytest.raises(SPTInvariantError, match="fit"):
            est.interventional_covariance({"X": 1.0})

    def test_empty_intervention_raises(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        with pytest.raises(SPTInvariantError, match="intervention"):
            est.interventional_covariance({})

    def test_ndarray_input_without_names_raises(self) -> None:
        data = np.random.default_rng(42).normal(size=(100, 3))
        est = CausalCovarianceEstimator(edges=[("X", "Y")])
        with pytest.raises(SPTInvariantError, match="variable_names"):
            est.fit(data)

    def test_ndarray_input_with_names(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        Y = 0.8 * X + rng.normal(size=n) * 0.3
        data = np.column_stack([X, Y])

        est = CausalCovarianceEstimator(edges=[("a", "b")])
        est.fit(data, variable_names=["a", "b"])
        cov = est.observational_covariance()
        assert cov.shape == (2, 2)


class TestCoreIntegration:
    """Integration with quantspt core covariance functions."""

    def test_output_usable_as_covariance_rate(
        self,
        known_sem_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        from quantspt.core.covariance import relative_covariance

        df, _ = known_sem_data
        est = CausalCovarianceEstimator(edges=[("X", "Y"), ("Y", "Z")])
        est.fit(df)
        cov = est.observational_covariance()

        pi = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        tau = relative_covariance(cov, pi)

        assert tau.shape == (3, 3)
        np.testing.assert_allclose(tau @ pi, 0.0, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(tau)
        assert np.all(eigenvalues >= -1e-10)
