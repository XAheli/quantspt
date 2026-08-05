"""Tests for statistical significance tests.

Validates bootstrap CI, permutation tests, and multiple testing
corrections against known analytical properties.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.statistical_tests import (
    BootstrapCIResult,
    PermutationTestResult,
    bootstrap_confidence_interval,
    multiple_testing_correction,
    permutation_test,
)

# =========================================================================
# Bootstrap confidence intervals
# =========================================================================


class TestBootstrapCI:
    """Tests for bootstrap confidence interval estimation."""

    def test_ci_contains_true_mean(self) -> None:
        """95% CI from normal data should contain the true mean."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, 1000)

        result = bootstrap_confidence_interval(
            data,
            statistic_func=np.mean,
            confidence_level=0.95,
            n_bootstrap=5000,
            rng=np.random.default_rng(123),
        )

        assert result.ci_lower < true_mean < result.ci_upper

    def test_ci_width_shrinks_with_more_data(self) -> None:
        """Larger sample → narrower CI."""
        rng = np.random.default_rng(42)

        small_data = rng.normal(0, 1, 50)
        large_data = rng.normal(0, 1, 500)

        ci_small = bootstrap_confidence_interval(
            small_data,
            np.mean,
            n_bootstrap=3000,
            rng=np.random.default_rng(1),
        )
        ci_large = bootstrap_confidence_interval(
            large_data,
            np.mean,
            n_bootstrap=3000,
            rng=np.random.default_rng(1),
        )

        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        assert width_large < width_small

    def test_higher_confidence_wider_ci(self) -> None:
        """99% CI should be wider than 90% CI."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)

        ci_90 = bootstrap_confidence_interval(
            data,
            np.mean,
            confidence_level=0.90,
            n_bootstrap=3000,
            rng=np.random.default_rng(1),
        )
        ci_99 = bootstrap_confidence_interval(
            data,
            np.mean,
            confidence_level=0.99,
            n_bootstrap=3000,
            rng=np.random.default_rng(1),
        )

        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_result_structure(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        result = bootstrap_confidence_interval(
            data,
            np.mean,
            n_bootstrap=1000,
            rng=np.random.default_rng(1),
        )

        assert isinstance(result, BootstrapCIResult)
        assert result.ci_lower < result.ci_upper
        assert result.confidence_level == 0.95
        assert result.n_bootstrap == 1000
        assert len(result.bootstrap_distribution) == 1000

    def test_point_estimate_matches_statistic(self) -> None:
        """Point estimate should be the statistic applied to original data."""
        rng = np.random.default_rng(42)
        data = rng.normal(3.0, 1.0, 200)

        result = bootstrap_confidence_interval(
            data,
            np.mean,
            n_bootstrap=1000,
            rng=np.random.default_rng(1),
        )
        assert_allclose(result.estimate, np.mean(data), atol=1e-14)

    def test_works_with_median(self) -> None:
        """Bootstrap works for non-mean statistics like median."""
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 500)

        result = bootstrap_confidence_interval(
            data,
            np.median,
            n_bootstrap=2000,
            rng=np.random.default_rng(1),
        )
        assert result.ci_lower < np.median(data) < result.ci_upper


# =========================================================================
# Permutation test
# =========================================================================


class TestPermutationTest:
    """Tests for the permutation test for strategy significance."""

    def test_identical_returns_not_significant(self) -> None:
        """When strategy = benchmark, should not be significant."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 200)

        result = permutation_test(
            returns,
            returns,
            n_permutations=1000,
            alpha=0.05,
            rng=np.random.default_rng(1),
        )
        assert result.p_value > 0.05
        assert result.significant is False

    def test_strong_outperformance_significant(self) -> None:
        """Clear outperformance should be significant."""
        rng = np.random.default_rng(42)
        strategy = rng.normal(0.005, 0.01, 200)
        benchmark = rng.normal(-0.005, 0.01, 200)

        result = permutation_test(
            strategy,
            benchmark,
            n_permutations=5000,
            alpha=0.05,
            rng=np.random.default_rng(1),
        )
        assert result.p_value < 0.05
        assert result.significant is True

    def test_p_value_in_zero_one(self) -> None:
        """p-value must be in [0, 1]."""
        rng = np.random.default_rng(42)
        strategy = rng.normal(0, 0.01, 100)
        benchmark = rng.normal(0, 0.01, 100)

        result = permutation_test(
            strategy,
            benchmark,
            n_permutations=500,
            rng=np.random.default_rng(1),
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_result_structure(self) -> None:
        rng = np.random.default_rng(42)
        s = rng.normal(0, 0.01, 50)
        b = rng.normal(0, 0.01, 50)

        result = permutation_test(
            s,
            b,
            n_permutations=500,
            alpha=0.10,
            rng=np.random.default_rng(1),
        )

        assert isinstance(result, PermutationTestResult)
        assert result.n_permutations == 500
        assert result.alpha == 0.10
        assert np.isfinite(result.observed_statistic)

    def test_observed_statistic_is_mean_excess(self) -> None:
        """Test statistic should be mean(strategy - benchmark)."""
        strategy = np.array([0.01, 0.02, 0.03, 0.04])
        benchmark = np.array([0.005, 0.01, 0.015, 0.02])

        result = permutation_test(
            strategy,
            benchmark,
            n_permutations=500,
            rng=np.random.default_rng(1),
        )
        expected = float(np.mean(strategy - benchmark))
        assert_allclose(result.observed_statistic, expected, atol=1e-14)


# =========================================================================
# Multiple testing correction
# =========================================================================


class TestMultipleTestingCorrection:
    """Tests for Bonferroni and FDR corrections."""

    def test_bonferroni_scales_by_m(self) -> None:
        """Bonferroni: adjusted = min(m * p, 1)."""
        p_values = np.array([0.01, 0.04, 0.05])
        adjusted, sig = multiple_testing_correction(
            p_values, method="bonferroni", alpha=0.05
        )
        assert_allclose(adjusted[0], 0.03)
        assert_allclose(adjusted[1], 0.12)
        assert_allclose(adjusted[2], 0.15)

    def test_bonferroni_capped_at_one(self) -> None:
        """Adjusted p-values cannot exceed 1."""
        p_values = np.array([0.5, 0.8])
        adjusted, _ = multiple_testing_correction(p_values, method="bonferroni")
        assert np.all(adjusted <= 1.0)

    def test_bonferroni_significance(self) -> None:
        """Only p * m < alpha should be significant."""
        p_values = np.array([0.01, 0.03, 0.10])
        _, sig = multiple_testing_correction(p_values, method="bonferroni", alpha=0.05)
        assert sig[0] is np.True_  # 0.01 * 3 = 0.03 < 0.05
        assert sig[1] is np.False_  # 0.03 * 3 = 0.09 > 0.05
        assert sig[2] is np.False_  # 0.10 * 3 = 0.30 > 0.05

    def test_fdr_less_conservative_than_bonferroni(self) -> None:
        """FDR correction should reject at least as many as Bonferroni."""
        p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05])
        _, sig_bonf = multiple_testing_correction(
            p_values, method="bonferroni", alpha=0.05
        )
        _, sig_fdr = multiple_testing_correction(p_values, method="fdr", alpha=0.05)
        assert np.sum(sig_fdr) >= np.sum(sig_bonf)

    def test_fdr_adjusted_monotone(self) -> None:
        """Sorted adjusted p-values should be non-decreasing."""
        p_values = np.array([0.001, 0.01, 0.03, 0.10, 0.20])
        adjusted, _ = multiple_testing_correction(p_values, method="fdr", alpha=0.05)
        sorted_adj = np.sort(adjusted)
        assert np.all(np.diff(sorted_adj) >= -1e-14)

    def test_single_pvalue(self) -> None:
        """With m=1, adjusted = original for both methods."""
        p_values = np.array([0.03])
        adj_bonf, _ = multiple_testing_correction(p_values, method="bonferroni")
        adj_fdr, _ = multiple_testing_correction(p_values, method="fdr")
        assert_allclose(adj_bonf[0], 0.03, atol=1e-14)
        assert_allclose(adj_fdr[0], 0.03, atol=1e-14)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            multiple_testing_correction(
                np.array([0.05]),
                method="invalid",  # type: ignore[arg-type]
            )
