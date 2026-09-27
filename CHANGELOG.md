# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold with full package structure
- `SPTResult` envelope class for consistent return types
- `require()` / `ensure()` precondition/postcondition helpers
- Rich error hierarchy (`SPTError`, `SPTInvariantError`, `OptimizationError`, …)
- Type aliases and protocols (`StochasticProcess`, `Discretization`, `PortfolioGenerator`)
- Plugin system with entry-point discovery
- `pyproject.toml` with core, viz, opt, sim, gpu, data, and dev extras
- CI pipeline scaffold (GitHub Actions)

### Fixed

- `experimental.neural_fgp.NeuralFGP.fit` stored the caller's
  `NeuralFGPConfig` by reference and then overwrote `positivity_offset` during
  warm-start calibration, so a user-supplied offset silently changed on the
  caller's own object. The model now keeps a private copy of the config.
- `universe.criteria.pairwise_correlation_score`: zeroing the correlation
  diagonal wrote into `DataFrame.values`, which pandas >= 3 exposes as a
  read-only buffer, so the function (and every `SPTUniverseSelector` /
  `select_timeseries` call that uses it) raised `ValueError: underlying array
  is read-only`. The diagonal is now zeroed on an explicit writable copy.
- `network.topology.build_granger_network`: the `verbose=False` keyword was
  removed in statsmodels >= 0.15, so every pairwise test raised `TypeError`,
  which the surrounding handler recorded as a warning — the function then
  returned an empty network (`n_edges=0`) that looked like "no causality
  detected". The call is now compatible with statsmodels >= 0.15.
- `data.corporate_actions.adjust_for_dividends`: the `total_return` method
  inverted the adjustment direction — pre-dividend prices were scaled *up* by
  `1/(1-d/p)` rather than down by the CRSP factor `1-d/p`, moving historical
  prices away from the true total-return level. `total_return` and
  `proportional` now share one cumulative CRSP implementation and both
  back-adjust pre-dividend prices downward.
- `require()` / `ensure()` now accept NumPy booleans in their type annotations,
  removing four `mypy` `arg-type` errors at existing call sites.
- `detect_splits()` narrows array indices to `int` before indexing a pandas
  `Index`, fixing a `mypy` `call-overload` error.
- Added `arch` and `statsmodels` to the `mypy` `ignore_missing_imports` list
  (optional and transitively-installed imports), so `mypy quantspt/` is clean.
- `visualization.model_diagnostics.plot_residuals` no longer raises NumPy's
  "Degrees of freedom <= 0" warning (promoted to an error by the test
  configuration) on 1-element input; the ±2σ bands collapse to the residual.
- `post_processing.clean_weights.enforce_bounds` docstring now states the real
  contract — bounded renormalisation: bounds respected, unit sum, idempotent.
  The previous "clip weights and renormalise" wording described a one-step
  algorithm whose output can violate the very bounds being enforced
  (e.g. `[0.6, 0.3, 0.1]` with `upper=0.5` rescales to `[0.556, …]`).
- Test configuration ignores hmmlearn's NumPy 2.5 `ndarray.shape` deprecation
  warning, which `filterwarnings = ["error"]` promoted to 13 spurious failures.
