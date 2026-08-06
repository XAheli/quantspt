"""Tests for regime detection — HMM and changepoint methods.

Validates that regime detectors correctly identify synthetic regimes
and integrate with the SPT diversity condition framework.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")
pytest.importorskip("ruptures")

from quantspt.ml.regime import ChangepointDetector, HMMRegimeDetector


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def two_regime_data(rng: np.random.Generator) -> np.ndarray:
    """Synthetic 2-regime time series: regime 0 (low mean) and regime 1 (high mean)."""
    regime_0 = rng.normal(loc=-1.0, scale=0.3, size=(200, 2))
    regime_1 = rng.normal(loc=2.0, scale=0.3, size=(200, 2))
    return np.vstack([regime_0, regime_1]).astype(np.float64)


@pytest.fixture
def changepoint_signal(rng: np.random.Generator) -> np.ndarray:
    """Signal with a clear changepoint at index 150."""
    seg1 = rng.normal(loc=0.0, scale=0.5, size=150)
    seg2 = rng.normal(loc=5.0, scale=0.5, size=150)
    return np.concatenate([seg1, seg2]).astype(np.float64)


# ---------------------------------------------------------------------------
# HMM Regime Detection Tests
# ---------------------------------------------------------------------------


class TestHMMRegimeDetector:
    """Tests for HMMRegimeDetector."""

    def test_fit_and_predict(self, two_regime_data: np.ndarray) -> None:
        """HMM should recover 2 regimes from clearly separated data."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(two_regime_data)
        labels = detector.predict(two_regime_data)

        assert labels.shape == (400,)
        assert set(np.unique(labels)) == {0, 1}

        first_half_label = labels[0]
        second_half_label = labels[399]
        assert first_half_label != second_half_label, (
            "HMM should detect different regimes for clearly separated data"
        )

    def test_recovers_correct_regime_labels(self, two_regime_data: np.ndarray) -> None:
        """Most labels in first half should match, same for second half."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(two_regime_data)
        labels = detector.predict(two_regime_data)

        first_half_mode = int(np.bincount(labels[:200]).argmax())
        second_half_mode = int(np.bincount(labels[200:]).argmax())
        assert first_half_mode != second_half_mode

        first_half_accuracy = (labels[:200] == first_half_mode).mean()
        second_half_accuracy = (labels[200:] == second_half_mode).mean()
        assert first_half_accuracy > 0.85, (
            f"First half accuracy: {first_half_accuracy:.2f}"
        )
        assert second_half_accuracy > 0.85, (
            f"Second half accuracy: {second_half_accuracy:.2f}"
        )

    def test_transition_matrix_is_stochastic(self, two_regime_data: np.ndarray) -> None:
        """Transition matrix rows must sum to 1."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(two_regime_data)
        P = detector.transition_matrix
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0)

    def test_predict_proba_shape(self, two_regime_data: np.ndarray) -> None:
        """predict_proba returns (T, n_regimes) probabilities."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(two_regime_data)
        proba = detector.predict_proba(two_regime_data)
        assert proba.shape == (400, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_forecast_diversity(self, two_regime_data: np.ndarray) -> None:
        """forecast_diversity returns reasonable multi-step forecasts."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(two_regime_data)
        forecasts = detector.forecast_diversity(horizon=10)
        assert forecasts.shape == (10, 2)
        np.testing.assert_allclose(forecasts.sum(axis=1), 1.0, atol=1e-6)

    def test_1d_features_work(self, rng: np.random.Generator) -> None:
        """Detector handles 1D feature input."""
        data = np.concatenate(
            [
                rng.normal(-2, 0.3, 100),
                rng.normal(2, 0.3, 100),
            ]
        )
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(data)
        labels = detector.predict(data)
        assert labels.shape == (200,)

    def test_unfitted_raises(self) -> None:
        """Accessing results before fit raises RuntimeError."""
        detector = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            detector.predict(np.ones((10, 2)))
        with pytest.raises(RuntimeError, match="fitted"):
            _ = detector.transition_matrix

    def test_n_regimes_property(self) -> None:
        """n_regimes property returns configured value."""
        detector = HMMRegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3


# ---------------------------------------------------------------------------
# Changepoint Detection Tests
# ---------------------------------------------------------------------------


class TestChangepointDetector:
    """Tests for ChangepointDetector."""

    def test_finds_known_breakpoint(self, changepoint_signal: np.ndarray) -> None:
        """Detector finds the changepoint near index 150."""
        detector = ChangepointDetector(model="l2", penalty=3.0, min_size=20)
        detector.fit(changepoint_signal)

        assert len(detector.changepoints) >= 1
        closest = min(detector.changepoints, key=lambda cp: abs(cp - 150))
        assert abs(closest - 150) < 30, (
            f"Detected changepoint at {closest}, expected near 150"
        )

    def test_predict_labels_segmented(self, changepoint_signal: np.ndarray) -> None:
        """predict() returns segment labels."""
        detector = ChangepointDetector(model="l2", penalty=3.0, min_size=20)
        detector.fit(changepoint_signal)
        labels = detector.predict(changepoint_signal)

        assert labels.shape == (300,)
        assert len(np.unique(labels)) >= 2

    def test_no_changepoints_for_stationary_signal(
        self, rng: np.random.Generator
    ) -> None:
        """Stationary signal should have few/no changepoints with high penalty."""
        signal = rng.normal(0, 1, size=200).astype(np.float64)
        detector = ChangepointDetector(model="l2", penalty=100.0, min_size=20)
        detector.fit(signal)
        assert len(detector.changepoints) <= 1

    def test_multiple_changepoints(self, rng: np.random.Generator) -> None:
        """Detector finds multiple changepoints in multi-regime signal."""
        seg1 = rng.normal(0, 0.3, 100)
        seg2 = rng.normal(5, 0.3, 100)
        seg3 = rng.normal(-3, 0.3, 100)
        signal = np.concatenate([seg1, seg2, seg3]).astype(np.float64)

        detector = ChangepointDetector(model="l2", penalty=3.0, min_size=20)
        detector.fit(signal)
        assert len(detector.changepoints) >= 2

    def test_multidimensional_signal(self, rng: np.random.Generator) -> None:
        """Detector handles multidimensional signals."""
        seg1 = rng.normal(0, 0.3, (100, 3))
        seg2 = rng.normal(5, 0.3, (100, 3))
        signal = np.vstack([seg1, seg2]).astype(np.float64)

        detector = ChangepointDetector(model="l2", penalty=3.0, min_size=20)
        detector.fit(signal)
        assert len(detector.changepoints) >= 1
