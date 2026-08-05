"""Tests for estimation/calibration -- Atlas model calibration.

Validates that calibration recovers original parameters from data
simulated by a known Atlas model, and that goodness-of-fit metrics
are well-behaved.

Mathematical References
-----------------------
- Atlas dynamics: BFK Eq. 1.1, 1.6-1.7
- Stability condition: BFK Eq. 1.5
- Pareto exponents: BFK Eq. 4.3-4.4
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.estimation.calibration import calibrate_atlas, goodness_of_fit
from quantspt.models.atlas import AtlasModel


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _simulate_atlas_market_caps(
    model: AtlasModel,
    T: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate Atlas model market caps via Euler-Maruyama."""
    n = model.n
    log_caps = np.log(np.ones(n) * 100.0)
    all_log_caps = [log_caps.copy()]

    for _ in range(T - 1):
        order = np.argsort(-log_caps)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n)

        drift = np.array([model.gamma + model.g[r] for r in ranks])
        sigma_vec = np.array([model.sigma[r] for r in ranks])
        dw = rng.standard_normal(n) * np.sqrt(dt)
        log_caps = log_caps + drift * dt + sigma_vec * dw
        all_log_caps.append(log_caps.copy())

    return np.exp(np.array(all_log_caps))


# =========================================================================
# A. Atlas Calibration
# =========================================================================


class TestCalibrateAtlas:
    """Tests for calibrate_atlas()."""

    def test_recovers_n_stocks(self, rng: np.random.Generator) -> None:
        """Calibration should correctly identify number of stocks."""
        model = AtlasModel(n=5, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)
        assert result["n"] == 5

    def test_stability_condition_satisfied(self, rng: np.random.Generator) -> None:
        """Calibrated g must satisfy stability (BFK Eq. 1.5)."""
        model = AtlasModel(n=4, gamma=0.05, g_param=0.02, sigma_param=0.25)
        caps = _simulate_atlas_market_caps(model, T=3000, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)

        g = result["g"]
        cumsum = np.cumsum(g)
        assert np.all(cumsum[:-1] < 0), f"Stability violated: {cumsum[:-1]}"
        assert_allclose(cumsum[-1], 0.0, atol=1e-8)

    def test_sigma_positive(self, rng: np.random.Generator) -> None:
        """All calibrated volatilities must be positive."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)
        assert np.all(result["sigma"] > 0)

    def test_pareto_exponents_positive(self, rng: np.random.Generator) -> None:
        """Pareto exponents must be positive."""
        model = AtlasModel(n=4, gamma=0.05, g_param=0.015, sigma_param=0.2)
        caps = _simulate_atlas_market_caps(model, T=3000, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)
        assert np.all(result["pareto_exponents"] > 0)

    def test_recovers_volatility_order_of_magnitude(
        self, rng: np.random.Generator
    ) -> None:
        """Calibrated sigma should be in the right ballpark."""
        true_sigma = 0.3
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=true_sigma)
        caps = _simulate_atlas_market_caps(model, T=5000, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)
        mean_sigma = float(np.mean(result["sigma"]))
        assert 0.1 < mean_sigma < 1.0

    def test_min_observations_check(self) -> None:
        caps = np.ones((10, 3)) * 100
        with pytest.raises(SPTInvariantError, match="observations"):
            calibrate_atlas(caps, min_observations=50)

    def test_rejects_negative_caps(self, rng: np.random.Generator) -> None:
        caps = rng.standard_normal((100, 3))
        with pytest.raises(SPTInvariantError, match="positive"):
            calibrate_atlas(caps)

    def test_rejects_single_stock(self) -> None:
        caps = np.ones((100, 1)) * 100
        with pytest.raises(SPTInvariantError, match="2 stocks"):
            calibrate_atlas(caps)

    def test_n_observations_stored(self, rng: np.random.Generator) -> None:
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=500, dt=1 / 252, rng=rng)
        result = calibrate_atlas(caps)
        assert result["n_observations"] == 500


# =========================================================================
# B. Goodness of Fit
# =========================================================================


class TestGoodnessOfFit:
    """Tests for goodness_of_fit()."""

    def test_ks_pvalues_shape(self, rng: np.random.Generator) -> None:
        model = AtlasModel(n=4, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        assert gof["ks_pvalues"].shape == (3,)

    def test_ks_pvalues_in_range(self, rng: np.random.Generator) -> None:
        """KS p-values should be in [0, 1]."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        valid = gof["ks_pvalues"][~np.isnan(gof["ks_pvalues"])]
        assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_capital_curve_rmse_non_negative(self, rng: np.random.Generator) -> None:
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        assert gof["capital_curve_rmse"] >= 0

    def test_ergodic_deviation_shape(self, rng: np.random.Generator) -> None:
        n = 4
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        assert gof["ergodic_deviation"].shape == (n,)

    def test_ergodic_deviation_non_negative(self, rng: np.random.Generator) -> None:
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=2000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        assert np.all(gof["ergodic_deviation"] >= 0)

    def test_good_fit_for_atlas_data(self, rng: np.random.Generator) -> None:
        """RMSE should be reasonable for data generated from the model."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.02, sigma_param=0.3)
        caps = _simulate_atlas_market_caps(model, T=5000, dt=1 / 252, rng=rng)
        params = calibrate_atlas(caps)
        gof = goodness_of_fit(caps, params)
        assert gof["capital_curve_rmse"] < 0.5
