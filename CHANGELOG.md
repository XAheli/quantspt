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
