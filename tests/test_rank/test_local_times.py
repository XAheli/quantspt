"""Tests for rank/local_times.py — local time estimation.

Validates:
- Analytical local time rates match BFK Eq. 3.7
- Atlas model local time rates: λ_{k,k+1} = 2kg
- Empirical local times are non-negative
- Stability condition enforcement
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.rank.local_times import (
    empirical_local_time_rates,
    empirical_local_times,
    local_time_rates_analytical,
    local_time_rates_atlas,
)


class TestAnalyticalLocalTimeRates:
    """λ_{k,k+1} = -2(g_1+...+g_k) from BFK Eq. 3.7."""

    def test_basic_atlas(self) -> None:
        n = 5
        g_param = 0.01
        g = np.full(n, -g_param)
        g[-1] = (n - 1) * g_param
        lam = local_time_rates_analytical(g)
        for k in range(n - 1):
            expected = 2.0 * (k + 1) * g_param
            np.testing.assert_allclose(lam[k], expected, atol=1e-14)

    def test_shape(self) -> None:
        g = np.array([-0.03, -0.02, -0.01, 0.06])
        lam = local_time_rates_analytical(g)
        assert lam.shape == (3,)

    def test_all_positive(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        lam = local_time_rates_analytical(g)
        assert np.all(lam > 0)

    def test_general_first_order(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        lam = local_time_rates_analytical(g)
        cumsum_g = np.cumsum(g)
        np.testing.assert_allclose(lam, -2.0 * cumsum_g[:-1])

    def test_rejects_unstable(self) -> None:
        g = np.array([0.02, -0.03, 0.01])
        with pytest.raises(Exception, match=r"[Ss]tability"):
            local_time_rates_analytical(g)

    def test_matches_atlas_model(self) -> None:
        from quantspt.models.atlas import AtlasModel

        n = 6
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=0.3)
        lam_model = model.local_time_rates()
        lam_func = local_time_rates_analytical(model.g)
        np.testing.assert_allclose(lam_func, lam_model, atol=1e-14)


class TestAtlasLocalTimeRates:
    """Specialised Atlas formula: λ_{k,k+1} = 2kg."""

    def test_formula(self) -> None:
        n = 4
        g_param = 0.02
        lam = local_time_rates_atlas(n, g_param)
        expected = 2.0 * g_param * np.arange(1, n)
        np.testing.assert_allclose(lam, expected)

    def test_shape(self) -> None:
        lam = local_time_rates_atlas(10, 0.01)
        assert lam.shape == (9,)

    def test_increasing(self) -> None:
        lam = local_time_rates_atlas(5, 0.03)
        assert np.all(np.diff(lam) > 0)

    def test_matches_analytical(self) -> None:
        n = 5
        g_param = 0.01
        g = np.full(n, -g_param)
        g[-1] = (n - 1) * g_param
        lam_atlas = local_time_rates_atlas(n, g_param)
        lam_analytical = local_time_rates_analytical(g)
        np.testing.assert_allclose(lam_atlas, lam_analytical, atol=1e-14)


class TestEmpiricalLocalTimes:
    """Empirical estimation must produce non-negative values."""

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        path = np.cumsum(rng.standard_normal((200, 3)), axis=0)
        lt = empirical_local_times(path, dt=0.01)
        assert np.all(lt >= 0)

    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        path = np.cumsum(rng.standard_normal((100, 5)), axis=0)
        lt = empirical_local_times(path, dt=0.01)
        assert lt.shape == (4,)

    def test_rates_shape(self) -> None:
        rng = np.random.default_rng(42)
        path = np.cumsum(rng.standard_normal((100, 4)), axis=0)
        rates = empirical_local_time_rates(path, dt=0.01)
        assert rates.shape == (3,)
        assert np.all(rates >= 0)

    def test_identical_paths_high_local_time(self) -> None:
        path = np.column_stack(
            [
                np.arange(100, dtype=np.float64),
                np.arange(100, dtype=np.float64) + 0.001,
            ]
        )
        lt = empirical_local_times(path, dt=0.01, epsilon=1.0)
        assert lt[0] > 0

    def test_well_separated_paths_low_local_time(self) -> None:
        t = np.arange(100, dtype=np.float64)
        path = np.column_stack([t + 100.0, t])
        lt = empirical_local_times(path, dt=0.01, epsilon=0.01)
        np.testing.assert_allclose(lt[0], 0.0, atol=1e-10)
