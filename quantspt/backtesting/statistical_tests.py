"""Statistical significance tests for backtested strategies.

Provides bootstrap confidence intervals, permutation tests, and
multiple testing corrections to assess whether observed outperformance
is statistically meaningful.

Mathematical References
-----------------------
- Bootstrap CI: Efron (1979), bias-corrected percentile method
- Permutation test: randomization inference for strategy significance
- Bonferroni correction: P_adj = min(m · P_i, 1)
- Benjamini-Hochberg FDR: controls false discovery rate at level q
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "BootstrapCIResult",
    "PermutationTestResult",
    "bootstrap_confidence_interval",
    "multiple_testing_correction",
    "permutation_test",
]


@dataclass(frozen=True)
class BootstrapCIResult:
    """Result of bootstrap confidence interval estimation.

    Attributes
    ----------
    estimate : float
        Point estimate (sample statistic).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    confidence_level : float
        Confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap samples used.
    bootstrap_distribution : NDArray[np.float64]
        Full bootstrap distribution of the statistic.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    bootstrap_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a permutation test for strategy significance.

    Attributes
    ----------
    observed_statistic : float
        The observed test statistic.
    p_value : float
        Proportion of permutations with statistic ≥ observed.
    n_permutations : int
        Number of permutations performed.
    significant : bool
        Whether the result is significant at the given alpha.
    alpha : float
        Significance level used.
    """

    observed_statistic: float
    p_value: float
    n_permutations: int
    significant: bool
    alpha: float


def bootstrap_confidence_interval(
    data: NDArray[np.float64],
    statistic_func: Callable[[NDArray[np.float64]], float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> BootstrapCIResult:
    """Compute bootstrap confidence interval for a statistic.

    Uses the percentile method: the CI bounds are the α/2 and 1-α/2
    percentiles of the bootstrap distribution.

    Parameters
    ----------
    data : ndarray
        The data to resample from.
    statistic_func : callable
        Function that computes the statistic from a data sample.
        Signature: statistic_func(data) → float.
    confidence_level : float
        Confidence level, e.g. 0.95 for 95% CI.
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    BootstrapCIResult
        Point estimate, CI bounds, and bootstrap distribution.
    """
    require(0.0 < confidence_level < 1.0, "confidence_level must be in (0, 1)")
    require(n_bootstrap > 0, "n_bootstrap must be positive")
    require(len(data) > 0, "data must be non-empty")

    if rng is None:
        rng = np.random.default_rng()

    point_estimate = float(statistic_func(data))
    n = len(data)

    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample = data[indices]
        bootstrap_stats[i] = statistic_func(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return BootstrapCIResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=bootstrap_stats,
    )


def permutation_test(
    strategy_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Permutation test for strategy outperformance significance.

    Tests H₀: strategy returns are exchangeable with benchmark returns.
    The test statistic is the mean difference in returns.

    Parameters
    ----------
    strategy_returns : ndarray of shape (T,)
        Strategy period returns.
    benchmark_returns : ndarray of shape (T,)
        Benchmark period returns.
    n_permutations : int
        Number of random permutations.
    alpha : float
        Significance level.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    PermutationTestResult
        Test statistic, p-value, and significance assessment.
    """
    require(
        len(strategy_returns) == len(benchmark_returns),
        "strategy and benchmark returns must have same length",
    )
    require(n_permutations > 0, "n_permutations must be positive")
    require(0.0 < alpha < 1.0, "alpha must be in (0, 1)")

    if rng is None:
        rng = np.random.default_rng()

    excess = strategy_returns - benchmark_returns
    observed = float(np.mean(excess))
    n = len(excess)

    count_ge = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        perm_stat = float(np.mean(signs * excess))
        if perm_stat >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        significant=p_value < alpha,
        alpha=alpha,
    )


def multiple_testing_correction(
    p_values: NDArray[np.float64],
    method: Literal["bonferroni", "fdr"] = "bonferroni",
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Apply multiple testing correction to a set of p-values.

    Parameters
    ----------
    p_values : ndarray of shape (m,)
        Raw p-values from multiple tests.
    method : {'bonferroni', 'fdr'}
        Correction method.
        - 'bonferroni': Bonferroni correction (FWER control).
        - 'fdr': Benjamini-Hochberg procedure (FDR control).
    alpha : float
        Significance level.

    Returns
    -------
    tuple of (adjusted_pvalues, significant)
        adjusted_pvalues : ndarray of shape (m,)
            Corrected p-values.
        significant : ndarray of shape (m,), dtype bool
            Whether each test is significant after correction.
    """
    require(len(p_values) > 0, "p_values must be non-empty")
    require(0.0 < alpha < 1.0, "alpha must be in (0, 1)")
    m = len(p_values)

    if method == "bonferroni":
        adjusted = np.minimum(p_values * m, 1.0)
        significant = adjusted < alpha

    elif method == "fdr":
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]
        adjusted = np.zeros(m)

        for i in range(m):
            rank = i + 1
            adjusted_val = sorted_pvals[i] * m / rank
            adjusted[i] = min(adjusted_val, 1.0)

        for i in range(m - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        result_adjusted = np.zeros(m)
        result_adjusted[sorted_indices] = adjusted
        adjusted = result_adjusted
        significant = adjusted < alpha

    else:
        msg = f"Unknown method '{method}', use 'bonferroni' or 'fdr'"
        raise ValueError(msg)

    return adjusted, np.asarray(significant, dtype=np.bool_)
