# var-model

This is an Ensemble Value-at-Risk and Expected Shortfall for a small equity portfolio.
It leverages the three main methods of calculating VaR independently:
- Historical
- Parametric (Variance-Covariance)
- Monte Carlo
The divergence of these methods is intended to the headline analysis for this repo.

Real market data is acquired through the Alpha Vantage API. Raw prices and
computed returns are then stored in a SQL database (via SQLAlchemy), and risk
results are written back to the same database for comparison across trials.

## Configuration

Alpha Vantage requires an API key. Copy `.env.example` to `.env` and set your
key — `.env` is git-ignored and must never be committed:

```
ALPHAVANTAGE_API_KEY=your_key_here
```

> The free tier allows ~5 requests/minute and ~25/day, so the data layer
> throttles requests and caches fetched data to the database.

> yFinance is a suitable alternative should request limits become a problem. However,
 due to the nature of yFinance being a scraper, Yahoo changing their website endpoints
 temporarily breaks the package. So, AlphaVantage's reliability was more important than unlimited
 pulls.

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
2. Database schema + Alpha Vantage ingestion.
3. VaR core: historical, variance-covariance, Monte Carlo.
4. Expected Shortfall.
5. Divergence analysis.
6. Final testing, cleanup, documentation.
