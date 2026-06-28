# Design Notes

This document records the *why* behind `var-model` — the design decisions that
aren't obvious from the code, with particular emphasis (in later phases) on the
divergence analysis that compares the three VaR methods.

Each phase appends its own section as its logic lands.

**Status at a glance:**

| Phase | Scope | State |
|---|---|---|
| 1 | Project scaffold (uv, src layout, CI, tooling) | done |
| 2 | Historical, parametric, Monte Carlo VaR + Expected Shortfall | done |
| 3 | Divergence analysis (ensemble/comparison) + SQL persistence | done |
| 4 | Alpha Vantage ingestion, returns/portfolio, end-to-end CLI | done |

---

## Phase 1 — Project scaffold

**Goal:** a packaged, tested, CI-backed skeleton that the domain code plugs into,
following a `src`-layout numerical-library template.

**Decisions:**

- **`uv` + `src` layout.** The package lives under `src/var_model/` so it imports
  only when installed — no `sys.path` hacks, and tests run against the installed
  package, catching packaging mistakes early. `uv` manages the environment and
  writes a committed `uv.lock` for reproducibility.
- **Build backend: `hatchling`.** `uv init` defaults to its own `uv_build`
  backend; we switched to `hatchling` for a more conventional, portable build
  that doesn't pin the build step to a specific `uv` version.
- **Single dependency set.** `numpy`, `scipy`, `pandas`, `sqlalchemy`,
  `requests`, and `python-dotenv` are all required runtime dependencies — one
  `uv sync` installs everything. *However,* the math modules (`var.py`,
  `risk.py`) are kept to `numpy`/`scipy` only at the code level so the risk core
  stays pure and its tests never depend on the network (`requests`) or a
  database (`sqlalchemy`). The separation is by module responsibility, not by
  package extras.
- **Tooling:** `ruff` (lint + import sort), `mypy` (types), `pytest` (tests),
  and a GitHub Actions pipeline that mirrors the local commands, so a green
  local run should mean a green CI run.

**Verification gate (all must pass before Phase 2):**

```
uv run ruff check .
uv run mypy
uv run pytest -q
uv run var-model --help
```

---

## Data source — Alpha Vantage (chosen over yfinance)

**Decision:** market data comes from the **Alpha Vantage** REST API, called
directly with `requests`, rather than `yfinance`.

**Why:**

- **A real, documented API with an explicit contract.** Alpha Vantage is a
  first-party data provider with a stable, authenticated REST interface;
  `yfinance` scrapes an undocumented Yahoo endpoint that breaks without notice.
- **`requests` over the `alpha_vantage` wrapper.** A single `GET` against a
  documented endpoint is transparent and dependency-light; the wrapper adds
  indirection without earning its keep for a focused tool.

**Consequences this forces on the data layer (Phase 2):**

- **Secret handling.** The key is read from the `ALPHAVANTAGE_API_KEY`
  environment variable (loaded from a git-ignored `.env` via `python-dotenv` in
  dev). It is never hard-coded, committed, or passed through the math core. A
  missing key raises a clear error. `.env.example` documents the requirement.
- **Rate limiting.** The free tier is ~5 requests/minute and ~25/day. Ingestion
  must throttle (≈12 s between calls) and cache fetched series to the database
  so a re-run doesn't re-spend the daily budget. This shapes the fetch design
  more than `yfinance` did.

---

## Methods — milestone 1: Historical VaR + Expected Shortfall

Each VaR method is built and committed independently. The historical method is
first because it is the most intuitive and serves as the sanity-check baseline
the other two are measured against.

**Shared API conventions (set here, used by all three methods):**

- All methods take the **same `returns` array** plus `confidence`, `horizon`,
  and `value`. Running every method on identical data is what makes the
  divergence analysis meaningful — differences are the *method*, not the input.
- `confidence ∈ (0, 1)`; tail probability `alpha = 1 - confidence`
  (`confidence=0.95` → the 5th percentile). Default is `0.95`.
- VaR and ES are returned as **positive losses** in the units of `value`
  (default `1.0`, so a loss *fraction*). This makes the `ES ≥ VaR` invariant
  read naturally.
- Multi-day horizons use the **square-root-of-time** rule.

**Historical VaR.** Sort the observed returns and read off the empirical
`alpha`-quantile — *no distributional assumption*. We use `numpy`'s linear
interpolation between order statistics (`np.quantile(..., method="linear")`).
The quantile is a return (typically negative); VaR is its negation.

**Historical Expected Shortfall.** Average every observed return at or below the
`alpha`-quantile — the mean loss *given* the threshold is breached. Because the
mean of the tail is at least as extreme as the threshold itself, `ES ≥ VaR`
holds by construction. `_quantile` is shared with VaR so the two metrics use the
*same* tail cutoff.

**Why this is the baseline.** Historical VaR's only assumption is that the
sampled past resembles the future. Its strength (no shape assumption — it
captures fat tails and skew if they are *in the window*) is also its weakness:
it is **hostage to its lookback window**. No crash in the sample → no crash in
the estimate, and the quantile is a step function of a finite sample (coarse and
jumpy in the deep tail). Those limitations are exactly what the parametric and
Monte Carlo methods trade against, and they set up the divergence discussion.

**Testing notes.** Reference values use a hand-built symmetric return series
whose quantile positions are integers, so the expected quantile lands exactly on
an order statistic with no interpolation. The independent cross-check draws a
large normal sample and compares historical VaR/ES to the closed-form normal
quantile and tail expectation (via `scipy`), with a method-aware relative
tolerance (5%) rather than exact equality — sampling noise and the empirical
quantile's coarseness make equality the wrong assertion.

---

## Methods — milestone 2: Parametric (variance-covariance)

The parametric method assumes returns are **normally distributed**, estimates
the mean and volatility from the sample, and reads VaR/ES off the closed-form
normal tail.

**Formulas** (with `z = Phi^{-1}(confidence)`, `phi` the standard-normal pdf,
`alpha = 1 - confidence`):

- `VaR = (z·sigma - mu) · sqrt(horizon) · value`
- `ES  = (sigma·phi(z)/alpha - mu) · sqrt(horizon) · value`

**Estimator choices and why:**

- **Unbiased volatility (`ddof=1`).** We are *estimating* the population sigma
  from a sample, so the sample standard deviation (dividing by `n-1`) is the
  right estimator. The reference tests pin this down by comparing against the
  exact `ddof=1` value.
- **The mean is estimated and subtracted, not assumed zero.** This is the key
  decision for the divergence analysis. Historical VaR already embeds the drift
  through the empirical quantile; if parametric assumed `mu = 0` while historical
  did not, the two would diverge for a reason that has nothing to do with
  distributional shape. By treating the mean identically, any remaining
  divergence is attributable to the **shape assumption** — which is exactly what
  we want to study. (For 1-day horizons `mu` is typically tiny next to `z·sigma`,
  so this rarely moves the number much, but it keeps the comparison honest.)
- **`scipy.stats.norm`** supplies `ppf`/`pdf`; `scipy` is allowed in the math
  core alongside `numpy`.

**The divergence story begins here.** On a genuinely normal sample the
parametric and historical numbers converge (the headline cross-check asserts
agreement within 5% on 200k–300k draws). They *diverge* on real returns because
real returns are **not** normal:

- **Fat tails (excess kurtosis).** Equity returns have more extreme moves than a
  normal allows. Parametric VaR, anchored to `z·sigma`, **understates** deep-tail
  risk; historical, reading an actual empirical quantile, captures those moves if
  they are in the window. The gap typically widens at 99% vs 95%.
- **Skew.** The normal is symmetric; real equity returns are often
  left-skewed (crashes bigger than melt-ups). Parametric cannot see this;
  historical can.
- **Where parametric wins.** It is smooth and stable (no dependence on whether a
  single bad day happens to sit in the window), and it extrapolates into the tail
  rather than being capped by the worst observation — so at very high confidence
  with a short window it can actually be the more *conservative* of the two.

Monte Carlo (milestone 3) will simulate from these same estimated parameters; on
a normal model it should converge to parametric, which makes it both a validation
of the parametric formula and the natural place to introduce a non-normal
generating distribution later.

---

## Methods — milestone 3: Monte Carlo

The Monte Carlo method estimates `mu`/`sigma` exactly as the parametric method
does, then **simulates** `n_sims` draws from `N(mu, sigma)` and reads VaR/ES off
the *simulated* distribution — empirical quantile for VaR, simulated-tail mean
for ES — reusing the same `_quantile`/tail machinery as the historical method.

**Design choices and why:**

- **Same estimator, simulated instead of solved.** Because MC draws from the
  fitted normal, on a normal model it must converge to the parametric closed
  form as `n_sims → ∞`. That convergence is the milestone's headline test (MC vs
  parametric within ~3–4% at 100k–200k draws) and it **validates two methods at
  once**: if the analytic formula and the simulation agree, both are almost
  certainly right.
- **Reproducibility via `seed`.** MC is stochastic, but a risk number you can't
  reproduce is hard to trust or store. A keyword-only `seed` makes a run
  repeatable; `risk_report` passes the *same* seed to MC VaR and ES so they draw
  from one simulation and the `ES ≥ VaR` invariant holds exactly rather than
  probabilistically.
- **`n_sims` default 100k.** Enough for a stable 99% tail while staying fast.
  It is validated (`>= 1`) in the MC branch, not in the shared `validate_inputs`,
  since it is method-specific — the same pattern as the parametric "≥ 2 returns"
  guard.
- **Horizon via the shared √-time rule.** MC simulates single-period returns and
  scales by `sqrt(horizon)` like the other two methods, keeping all three
  comparable. Simulating multi-step compounded paths is the obvious extension and
  the natural home for a non-normal step distribution.

**Where MC sits in the divergence story.** As currently built (Gaussian
generator) MC carries the **same normality assumption as parametric**, so it
shares parametric's blind spot to fat tails and skew and will likewise diverge
from historical on real returns. Its value here is twofold: (1) a cross-check
that the parametric algebra is correct, and (2) the one method whose *generating
distribution is a free parameter* — swapping the normal draw for a Student-t or a
bootstrap of the empirical returns would let MC capture fat tails while keeping
the smooth, extrapolating-into-the-tail character that historical lacks. That is
the most promising lever for the divergence analysis and a documented next step.

---

## Phase 2 complete — the three methods

All three VaR methods and their Expected Shortfall companions now exist in the
pure `numpy`/`scipy` core, each built and committed independently with a
five-pillar test suite, and bundled by `risk_report`. The divergence behaviour
is now demonstrable end to end: on normal data the three methods agree; on
fat-tailed data historical pulls away from parametric and (Gaussian) Monte Carlo
in the deep tail, with the gap widening at higher confidence and largest for ES.
The dedicated divergence-analysis tool and the data/persistence layer build on
this foundation.

---

## Phase 3 — milestone A: the divergence analysis (the headline)

`divergence.py` is where the three methods stop being parallel computations and
become a *comparison*. It is still pure `numpy`/`scipy`; it adds two things on
top of `risk_report`.

**1. Spread — how far apart are the methods?** For VaR and for ES,
`divergence_report` records the absolute spread (`max - min` across the three
methods) and a relative spread normalized by the most conservative estimate. The
relative spread is the single number that answers "do the methods agree here?" —
~0 means yes, and it grows precisely when the normal assumption fails.

**2. Diagnostics — *why* do they diverge?** This is the deliverable the project
is built around, so the explanation is computed, not just asserted:

- **Skewness.** The normal is symmetric; real equity returns are usually
  left-skewed. Parametric/MC cannot represent skew; historical can. Non-zero skew
  is one driver of divergence.
- **Excess kurtosis (Fisher).** A normal scores 0. Positive excess kurtosis =
  fat tails = more extreme moves than `z·sigma` admits. This is the *primary*
  driver: it is why historical VaR/ES exceed the normal-based methods at high
  confidence, and why the gap is largest for ES (which averages the very tail
  where fat tails dominate).
- **Jarque-Bera test.** Combines skew and kurtosis into one normality test. A
  p-value below `NORMALITY_ALPHA` (0.05) is the formal signal that parametric and
  Monte Carlo are operating outside their assumptions and should be read as
  *lower bounds* on tail risk. On genuinely normal data the test does not reject
  and the spread is small — the two facts move together, which is the thesis.

**Why the diagnostics live with the comparison, not the methods.** Each method is
deliberately ignorant of the others (that independence is what makes their
agreement meaningful). The judgement about *which* to trust, and why, belongs in
a layer above them — `divergence.py` — keeping the method cores clean.

**The causal story, stated once.** Methods agree ⇔ returns are ~normal (high JB
p-value, ~zero skew/excess-kurtosis, small spread). Methods diverge ⇔ returns are
skewed/fat-tailed (low JB p-value, positive excess kurtosis, large spread), and
the divergence is directional: **historical ≥ parametric ≈ Gaussian-MC** in the
tail. The test suite asserts both directions of this equivalence.

**Degenerate guard.** Zero-variance returns have no shape; diagnostics return
neutral values (zero skew/kurtosis, JB p-value 1.0) and the spread is zero rather
than dividing by zero.

---

## Phase 3 — milestone B: writing results back to SQL

The `data/` package is the I/O layer. It is the **only** part of the codebase
that imports SQLAlchemy; the math core never does, which is what keeps the core
testable without a database. Persistence consumes the plain result dicts the core
produces.

**Schema — normalized, comparison-friendly.** Two tables:

- `runs` — one row per analysis: the *parameters* it used (confidence, horizon,
  value, n_sims, seed, label), the *diagnostics* (mean, std, skew, excess
  kurtosis, Jarque-Bera + p-value), and the *spreads* (absolute and relative for
  VaR and ES). Everything needed to interpret a run lives on its row.
- `method_results` — three rows per run (historical, parametric, monte_carlo),
  each with that method's VaR and ES, under a `UNIQUE(run_id, method)`
  constraint so a method can't be double-recorded for a run.

Splitting per-method results into their own table (rather than twelve columns on
`runs`) means cross-run, cross-method queries are natural SQL — "show 99% ES by
method over the last N runs" is a simple join — which is the whole point of
persisting results: comparison over time.

**Compute/IO separation, with a convenience.** `save_divergence_report` persists
an *already-computed* dict, so computation and storage stay independent and each
is testable alone. `compute_and_save` is the end-to-end one-liner
(`divergence_report` → save) for the common case. The data layer may import the
core (`divergence`); the core never imports the data layer.

**Connection config.** `make_engine` resolves the database URL from an explicit
argument, then the `VAR_MODEL_DB_URL` environment variable, then a default local
SQLite file — the same env-first pattern used for the Alpha Vantage key. Tests
use in-memory SQLite, create the tables with `init_db`, and assert a true
round-trip by expiring the session so reads come from the database, not the
identity map.

**Scope note.** This stores *computed results*. Persisting raw prices and the
returns derived from them belongs with the Alpha Vantage ingestion layer (a later
phase); the schema here is deliberately about risk outputs, not market data.

---

## Phase 3 complete — ensemble, comparison, and persistence

The three methods are now pulled together by `divergence_report`, which quantifies
their spread and computes the diagnostics that explain it, and results are written
back to SQL for cross-run comparison. The headline deliverable — *why the methods
diverge* — is both computed (skew, kurtosis, Jarque-Bera) and documented, with the
test suite enforcing the causal equivalence in both directions. What remains is the
Alpha Vantage ingestion layer and wiring the CLI into the full
fetch → compute → persist pipeline.

---

## Phase 4 — ingestion, returns, and the end-to-end CLI

This closes the loop: real prices in, a stored, explained risk comparison out.

**Alpha Vantage ingestion (`data/fetch.py`).** The only network code in the
project. It calls the free `TIME_SERIES_DAILY` endpoint with `requests` and
returns closes as pandas Series.

- **Free endpoint, documented limitation.** The split/dividend-*adjusted*
  endpoint is premium, so we use raw `TIME_SERIES_DAILY`. Corporate actions are
  therefore not adjusted for; acceptable for a methods-comparison tool, and noted
  here rather than hidden.
- **Key handling.** Read from `ALPHAVANTAGE_API_KEY` (or passed explicitly),
  never hard-coded; a missing key raises a clear error.
- **Rate limiting.** Free tier is ~5 req/min, so multi-ticker fetches sleep ~12s
  between calls. Alpha Vantage signals throttling/errors in the JSON body (not
  HTTP status), so the parser surfaces `Note`/`Information`/`Error Message`
  verbatim as a `RuntimeError`.
- **Testability.** The HTTP client is injectable, so the whole layer is tested
  against canned payloads with no network and no rate-limit exposure.

**Returns and portfolio aggregation (`data/returns.py`).** Prices → per-asset
**log returns** → one **equal-weight** portfolio series. The portfolio return is
the weighted sum of asset log returns — the standard daily-horizon approximation
(exact under continuous compounding), which also keeps the portfolio a linear
combination of the assets, matching the variance-covariance mental model. This is
data-prep (pandas) and lives in the data layer, *not* the pure math core.

**Prices vs. returns persistence — single source of truth.** Both raw prices and
computed returns are cached (a stated project goal), but **prices are the source
of truth**: returns are derived from them and stored only as a convenience/audit
artifact. The unique `(ticker, date)` constraint on both tables makes ingestion
**idempotent** — a re-run upserts rather than duplicating, so it never re-spends
the API budget or corrupts the cache.

**Orchestration (`data/pipeline.py`).** `run_portfolio_analysis` is the single
place that spans both layers: fetch (or reuse cache) → save prices → log returns
→ save returns → equal-weight portfolio → trailing window → `compute_and_save`.
Keeping it here lets the CLI stay thin and keeps the math core ignorant of I/O.

**CLI (`cli.py`).** Two subcommands. `run` executes the pipeline and prints the
three-method comparison, the spread, and a plain-language normality verdict
(JB p < 0.05 → "methods expected to diverge"); `history` lists stored runs.
`--no-fetch` runs entirely off the cache (useful when the daily API budget is
spent), and `--db` selects the database. `.env` is loaded on startup so the key
is available. The CLI does no math — it parses, delegates to the pipeline, and
formats.

**Default portfolio (`config.py`).** AAPL / JPM / XOM / JNJ / PG — deliberately
cross-sector (tech, financials, energy, healthcare, staples) so the returns carry
varied tail behaviour and the divergence story has something to show. A ~504-day
(~2-year) trailing window gives the 99% historical tail enough observations.

---

## Phase 4 complete — the loop is closed

`var-model run` now goes from live tickers to a stored, explained risk
comparison; `var-model run --no-fetch` and `var-model history` work entirely off
the cache. Prices and returns are cached idempotently, results are persisted for
cross-run comparison, and the whole pipeline is tested without touching the
network. The pure `numpy`/`scipy` core remains free of any I/O.
