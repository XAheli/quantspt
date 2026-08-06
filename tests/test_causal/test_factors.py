"""Tests for causal factor models.

Validates B matrix extraction, noise covariance recovery, and the
distinction between causal factors (DAG roots) and assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.causal.factors import CausalFactorModel
from quantspt.errors import SPTInvariantError


@pytest.fixture()
def factor_asset_data() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Synthetic factor → asset data.

    True model: F1, F2 are factors (roots)
        A1 = 0.6*F1 + ε
        A2 = 0.4*F1 + 0.5*F2 + ε
        A3 = 0.7*F2 + ε
    """
    rng = np.random.default_rng(42)
    n = 1000

    F1 = rng.normal(size=n)
    F2 = rng.normal(size=n)
    A1 = 0.6 * F1 + rng.normal(scale=0.2, size=n)
    A2 = 0.4 * F1 + 0.5 * F2 + rng.normal(scale=0.15, size=n)
    A3 = 0.7 * F2 + rng.normal(scale=0.25, size=n)

    df = pd.DataFrame({"F1": F1, "F2": F2, "A1": A1, "A2": A2, "A3": A3})
    edges = [
        ("F1", "A1"),
        ("F1", "A2"),
        ("F2", "A2"),
        ("F2", "A3"),
    ]
    return df, edges


@pytest.fixture()
def chain_data() -> pd.DataFrame:
    """Chain graph X → Y → Z."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=n)
    Y = 0.8 * X + rng.normal(scale=0.3, size=n)
    Z = 0.5 * Y + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


class TestCausalFactorModel:
    """Core factor model tests."""

    def test_fit_returns_self(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        result = fm.fit(df)
        assert result is fm

    def test_factor_detection(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        """Root nodes (F1, F2) should be auto-detected as factors."""
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        assert set(fm.factor_names) == {"F1", "F2"}
        assert set(fm.asset_names) == {"A1", "A2", "A3"}

    def test_explicit_factor_names(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges, factor_names=["F1"])
        fm.fit(df)
        assert fm.factor_names == ["F1"]
        assert "F2" in fm.asset_names

    def test_causal_loadings_shape(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        B = fm.causal_loadings()
        assert B.shape == (5, 5)

    def test_factor_loadings_shape(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        """factor_loadings() returns assets × factors sub-matrix."""
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        fl = fm.factor_loadings()
        assert fl.shape == (3, 2)

    def test_causal_loadings_recover_true_values(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        B = fm.causal_loadings()
        names = fm.variable_names
        idx = {n: i for i, n in enumerate(names)}

        np.testing.assert_allclose(B[idx["A1"], idx["F1"]], 0.6, atol=0.05)
        np.testing.assert_allclose(B[idx["A2"], idx["F1"]], 0.4, atol=0.05)
        np.testing.assert_allclose(B[idx["A2"], idx["F2"]], 0.5, atol=0.05)
        np.testing.assert_allclose(B[idx["A3"], idx["F2"]], 0.7, atol=0.05)

        assert B[idx["F1"], idx["A1"]] == 0.0
        assert B[idx["F2"], idx["A3"]] == 0.0

    def test_noise_covariance_shape(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        omega = fm.noise_covariance()
        assert omega.shape == (5,)
        assert np.all(omega > 0)

    def test_noise_covariance_recovers_true_values(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        omega = fm.noise_covariance()
        names = fm.variable_names
        idx = {n: i for i, n in enumerate(names)}

        np.testing.assert_allclose(omega[idx["A1"]], 0.04, atol=0.02)
        np.testing.assert_allclose(omega[idx["A2"]], 0.0225, atol=0.02)
        np.testing.assert_allclose(omega[idx["A3"]], 0.0625, atol=0.02)


class TestReconstructCovariance:
    """Verify Σ = (I-B)^{-1} Ω (I-B)^{-T}."""

    def test_reconstruct_matches_observational(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        from quantspt.causal.covariance import CausalCovarianceEstimator

        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)

        cov_est = CausalCovarianceEstimator(edges=edges)
        cov_est.fit(df)
        obs_cov = cov_est.observational_covariance()
        reconstructed = fm.reconstruct_covariance()

        np.testing.assert_allclose(reconstructed, obs_cov, atol=1e-8)

    def test_reconstruct_is_symmetric_psd(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)
        cov = fm.reconstruct_covariance()

        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_direct_formula_verification(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        """Manually verify (I-B)^{-1} Ω (I-B)^{-T}."""
        df, edges = factor_asset_data
        fm = CausalFactorModel(edges=edges)
        fm.fit(df)

        B = fm.causal_loadings()
        omega = fm.noise_covariance()
        n = len(fm.variable_names)

        I_minus_B_inv = np.linalg.inv(np.eye(n) - B)
        expected = I_minus_B_inv @ np.diag(omega) @ I_minus_B_inv.T
        np.testing.assert_allclose(fm.reconstruct_covariance(), expected, atol=1e-10)


class TestFromPgmpy:
    """Pluggability: user provides their own LGBN."""

    def test_from_pgmpy(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        df, edges = factor_asset_data
        model = LinearGaussianBayesianNetwork(edges)
        model.fit(df)

        fm = CausalFactorModel.from_pgmpy(model)
        assert set(fm.factor_names) == {"F1", "F2"}
        B = fm.causal_loadings()
        assert B.shape == (5, 5)

    def test_from_pgmpy_with_explicit_factors(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        df, edges = factor_asset_data
        model = LinearGaussianBayesianNetwork(edges)
        model.fit(df)

        fm = CausalFactorModel.from_pgmpy(model, factor_names=["F1"])
        assert fm.factor_names == ["F1"]
        fl = fm.factor_loadings()
        assert fl.shape == (4, 1)

    def test_from_pgmpy_chain(self, chain_data: pd.DataFrame) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(chain_data)

        fm = CausalFactorModel.from_pgmpy(model)
        assert fm.factor_names == ["X"]
        assert set(fm.asset_names) == {"Y", "Z"}


class TestValidation:
    """Input validation and error handling."""

    def test_access_before_fit(self) -> None:
        fm = CausalFactorModel(edges=[("X", "Y")])
        with pytest.raises(SPTInvariantError, match="fit"):
            fm.causal_loadings()
        with pytest.raises(SPTInvariantError, match="fit"):
            fm.noise_covariance()
        with pytest.raises(SPTInvariantError, match="fit"):
            fm.reconstruct_covariance()
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = fm.variable_names
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = fm.factor_names

    def test_no_edges_raises(self) -> None:
        fm = CausalFactorModel()
        data = np.random.default_rng(42).normal(size=(100, 3))
        with pytest.raises(SPTInvariantError, match="edges"):
            fm.fit(data, variable_names=["X", "Y", "Z"])

    def test_edges_at_fit_time(
        self,
        factor_asset_data: tuple[pd.DataFrame, list[tuple[str, str]]],
    ) -> None:
        df, edges = factor_asset_data
        fm = CausalFactorModel()
        fm.fit(df, edges=edges)
        assert len(fm.factor_names) > 0

    def test_ndarray_input(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        Y = 0.8 * X + rng.normal(size=n) * 0.3
        data = np.column_stack([X, Y])

        fm = CausalFactorModel(edges=[("a", "b")])
        fm.fit(data, variable_names=["a", "b"])
        assert fm.factor_names == ["a"]
        assert fm.asset_names == ["b"]
