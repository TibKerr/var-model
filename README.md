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

### Python API

The library is complete for all three methods, Expected Shortfall, the
comparison layer, and SQL persistence:

```python
import numpy as np
from var_model import value_at_risk, risk_report, divergence_report

returns = np.random.default_rng(0).normal(0.0, 0.02, 1000)

# A single method
value_at_risk(returns, confidence=0.99, method="historical", value=1_000_000)

# VaR + ES across all three methods on the same data
risk_report(returns, confidence=0.99, value=1_000_000)

# Full comparison: the six numbers, the spread between methods, and the
# shape diagnostics (skew, excess kurtosis, Jarque-Bera) that explain why
# the methods diverge
divergence_report(returns, confidence=0.99, value=1_000_000)
```

Persist a run to SQL and read runs back:

```python
from sqlalchemy.orm import Session
from var_model.data import make_engine, init_db, compute_and_save, load_runs

engine = make_engine()  # VAR_MODEL_DB_URL env var, or a default local SQLite file
init_db(engine)
with Session(engine) as session:
    compute_and_save(session, returns, confidence=0.99, value=1_000_000, label="run-1")
    runs = load_runs(session)
```

### Command line

The CLI runs the full pipeline: fetch prices → compute returns → VaR/ES three
ways → store the result → print the comparison.

```
# Fetch the default portfolio (AAPL JPM XOM JNJ PG) and store a 99% run
uv run var-model run -c 0.99 --value 1000000 --label "first run"

# Re-run entirely from cached prices (no API call) on a custom portfolio
uv run var-model run --no-fetch --tickers AAPL MSFT --seed 0

# List previously stored runs
uv run var-model history
```

Fetching requires `ALPHAVANTAGE_API_KEY` in `.env` (see Configuration). Because
the free tier is limited to ~25 requests/day, `--no-fetch` reuses the cached
prices once you have pulled them.

## Project status

All phases complete: scaffold; historical, parametric, and Monte Carlo VaR + ES;
the divergence/comparison analysis; SQL persistence; Alpha Vantage ingestion and
the end-to-end CLI.

See `DESIGN.md` for the rationale behind each phase.

## Development

```
uv run ruff check .   # lint
uv run mypy           # type check
uv run pytest -q      # tests
```
