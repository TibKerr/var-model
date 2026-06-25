# var-model

Value-at-Risk and Expected Shortfall for a small equity portfolio, computed
three independent ways — **historical**, **variance-covariance (parametric)**,
and **Monte Carlo** — with the divergence between methods as the headline
analysis.

Real price data is pulled with `yfinance`, raw prices and computed returns are
persisted to SQL (SQLAlchemy), and risk results are written back to the same
database for comparison across runs.

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
