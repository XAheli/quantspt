# Contributing to quantspt

Thank you for your interest in contributing! This guide helps you get started.

## Contribution Levels

### Users

Install and use the library, file issues, ask questions, share feedback.

### Contributors

Fix bugs, add notebooks, improve documentation, submit data providers.
No deep SPT knowledge required — excellent Python skills are sufficient.

### Developers

Extend core math, add market models, change architecture.
Requires understanding of stochastic calculus and SPT.

## Development Setup

```bash
git clone https://github.com/XAheli/quantspt.git
cd quantspt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

- **Formatter**: `ruff format`
- **Linter**: `ruff check`
- **Type checker**: `mypy --strict`
- **Tests**: `pytest tests/`

All code must pass these checks before merge.

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure all checks pass: `make check`
4. Submit a PR with a clear description
5. Reference any related issues

## Testing Philosophy

- **Test LOC > source LOC** — tests are the specification
- **Every theorem → at least one test**
- **Property-based testing** for mathematical invariants
- **Golden regression tests** for numerical outputs
- **Never use bare `==` for floats** — use `np.isclose` or `quantspt.utils.close`

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):

```
feat(core): add entropy generating function
fix(simulation): correct Milstein scheme diffusion term
docs(tutorials): add Atlas model calibration notebook
test(arbitrage): add property test for mirror portfolio identity
```

## CI Notes

- CodeQL Analysis is managed via GitHub's default security settings, not a workflow file.
  To disable or configure, go to **Settings → Code security and analysis** in the GitHub UI.

## Areas for Contribution

| Area | Skills Needed | Level |
|------|--------------|-------|
| Core math | Stochastic calculus, SPT theory | Developer |
| Implementation | Python, NumPy, optimisation | Developer |
| Data providers | API integration, finance data | Contributor |
| Documentation | Technical writing, LaTeX | Contributor |
| Examples | Finance domain knowledge | Contributor |
| Testing | Property-based testing | Contributor |
| Visualisation | matplotlib, Plotly | Contributor |
