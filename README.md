# var-model

This is an Ensemble Value-at-Risk and Expected Shortfall for a small equity portfolio.
It leverages the three main methods of calculating VaR independently:
- Historical
- Parametric (Variance-Covariance)
- Monte Carlo
The divergence of these methods is intended to the headline analysis for this repo.

Real market data is acquired through yFinance. Raw prices and computed
returns are then stored in SQL databases (SQL Academy), and risk results are written back to'
the same database for comparison across trials.

## Install

```
uv sync
```

## Usage

```
uv run var-model --help
```

> The CLI grows as phases land (data fetch, VaR computation, divergence report).
> Phase 1 ships the runnable scaffold.

## Development

```
uv run ruff check .   # lint
uv run mypy           # type check
uv run pytest -q      # tests
```

## Project status

Built in discrete phases (see `DESIGN.md` for the rationale behind each):

1. **Scaffold** — uv, src layout, tooling, CI. ✅
2. Database schema + yfinance ingestion.
3. VaR core: historical, variance-covariance, Monte Carlo.
4. Expected Shortfall.
5. Divergence analysis.
6. Final testing, cleanup, documentation.
