# Design Notes

This document records the decisions behind this repository.

Each phase appends its own section as its logic lands.

| Phase | Scope | State |
|---|---|---|
| 1 | Project scaffold (uv, src layout, CI, tooling)
| 2 | Historical, parametric, Monte Carlo VaR + Expected Shortfall
| 3 | Divergence analysis (ensemble/comparison) + SQL persistence
| 3.5 | Closing the loop: Alpha Vantage ingestion, returns/portfolio, CLI
| 4 | Validation against real portfolio data

---

## Phase 1 — Project scaffold

**Goal:** a packaged, tested, Continuous Integration (CI)-backed skeleton that the domain code plugs into. It follows a standard `src`-layout numerical-library template.

**Decisions:**

- **`uv` + `src` layout.** The package exists as `src/var_model/`; it imports
  only when installed. So, there is no `sys.path` hacks, and tests run against the *installed*
  package. This catches packaging mistakes early. `uv` manages the environment and
  writes a committed `uv.lock` for reproducibility.
- **Build backend: `hatchling`.** `uv init` defaults to its own `uv_build`
  backend; I opted to use `hatchling` for a more conventional, portable build
  that won't pin the build step to a specific `uv` version.
- **Single dependency set.** `numpy`, `scipy`, `pandas`, `sqlalchemy`,
  `requests`, and `python-dotenv` are all required runtime dependencies. One
  `uv sync` installs everything. *However,* the math modules (`var.py`,
  `risk.py`) are kept to `numpy`/`scipy` only at the code level so the risk core
  stays pure. That way, its tests never depend on the network (`requests`) or a
  database (`sqlalchemy`). The separation is by module responsibility, not by
  package extras.
- **Tooling:** `ruff` (lint + import sort), `mypy` (types), `pytest` (tests),
  and a GitHub Actions pipeline that mirrors the local commands. This way, a green
  local run should mean a green CI run.

**Verification gate:**

```
uv run ruff check .
uv run mypy
uv run pytest -q
uv run var-model --help
```

Do *NOT* move on until each check passes.

---

## Data source — Alpha Vantage (chosen over yfinance)

**Decision:** market data comes from the **Alpha Vantage** REST API, called
directly with `requests`, rather than `yfinance`.

**Reasoning**

- Alpha Vantage is a first-party data provider with a stable, authenticated REST interface;
  `yfinance` scrapes an undocumented Yahoo endpoint that breaks without notice. The compromise
  of losing the ease of implementation is less important than having a reliable engine.
- A single `GET` against a documented endpoint is transparent and dependency-light; the wrapper adds
  indirection without earning its keep for a focused tool.
- Because data is aggregated in the SQL database (discussed later), the 100 day lookback is no longer a
  limitation of the AlphaVantage API.

**Consequences:**

- Like with any API, the key is read from the `ALPHAVANTAGE_API_KEY`
  environment variable (loaded from a git-ignored `.env` via `python-dotenv` in
  dev). It is never hard-coded, committed, or passed through the math core. A
  missing key raises a clear error. `.env.example` documents the requirement for
  reproducibility.
- The free tier is ~5 requests/minute and ~25/day. Ingestion
  must throttle (≈12 s between calls) and cache fetched series to the database
  so a re-run doesn't re-spend the daily budget. This shapes the fetch design
  more than `yfinance` did.

---

## Methods: Core Concepts

**Shared API Conventions:**

All methods accept the same `returns` array plus `confidence`, `horizon`, and `value`. Running
every method on identical inputs is what makes the divergence analysis meaningful. The differences are meant to reflect the method, not the data.

- $\text{confidence} \in (0,1)$; tail probability $\text{alpha} = 1 - \text{confidence}$    
  ($\text{confidence} = .95 \rightarrow$ the 5th percentile). Default is $.95$
- VaR and ES are returned as positive losses in the units of $\text{value}$ (default is $1.0$,
  so a loss fraction)
- Multi-day horizons use the square-root-of-time rule (e.g. for a 7 day time period, we multiply the daily volatility by $\sqrt{7}$)

**Expected Shortfall:**

Expected Shortfall (ES) is the mean loss *given* that the VaR threshold has been breached. It does not just provide threshold, but the magnitude beyond the threshold. ES is computed identically across each method: average every return at or below the $\text{alpha}$-quantile.

**ES $\geq$ VaR Invariant:**

By definition, the mean of the tail is at least as extreme as the threshold that defines it, so ES $\geq$ VaR holds as a mathematical property across all three methods. Each method has its own implementation for enforcing this invariant, documented in its respective Design Choices section.

**Data ingestion and Caching:**

Market data is fetched from the Alpha Vantage `TIME_SERIES_DAILY` endpoint. The free tier returns only the last ~100 trading days per request (`outputsize='compact'`), which is sufficient for a stable 95% estimate. But, it makes the 99% historical tail coarse; that is, it leans on the worst one or two observations in the window.

The data layer bypasses this limit through idempotent accumulation. Fetched prices are upserted into a local SQLite database keyed on `(ticker, date)`: existing rows are updated in place and new rows are appended, so no date is ever duplicated and no API budget is re-spent on data already held. All three methods then consume prices loaded from this cache rather than directly from the API. Running the pipeline regularly therefore builds a historical window that grows beyond 100 points over time without requiring a premium key. A `--no-fetch` flag allows re-running the analysis entirely from cached data once prices have been pulled.

Alternatively, `outputsize='full'` with a premium key fetches the complete available history in a single request.

---

## Methods: Historical VaR

**Overview:**

Historical VaR sorts observed returns and reads off the empirical `alpha`-quantile with no distributional assumption. Its only assumption is that the sampled past resembles the future, which makes it the most intuitive method and the natural baseline against which the parametric and Monte Carlo methods are measured.

**Formulas:**

Let $r_{\alpha}$ denote the empirical `alpha`-quantile of the return series, computed via linear interpolation between order statistics.

  $VaR = -r_{\alpha} \cdot \sqrt{h} \cdot V$

  $ES = - \frac{1}{|\mathcal{T}|} \sum_{r_{i} \in \mathcal{T}}r_{t} \cdot \sqrt{h} \cdot V$

where $\mathcal{T} = \{r_t : r_t \leq r_{\alpha}\}$ is the tail set, $h$ is the horizon in days, and $V$ is the portfolio value.

**Design Choices:**

- Linear Interpolation: (`np.quantile(...,method='linear')`). Interpolates between order statistics rather than snapping to the nearest observed value, producing a smoother quantile estimate across confidence levels.
- Shared `_quantile` for VaR and ES. Both metrics use the same tail cutoff, which is how the implementation enforces the $ES \geq VaR$ invariant for this method.

**Strengths and Limitations:**

*Strengths:* Makes no shape assumption — fat tails, skew, and multimodality are all captured if they appear in the window. Intuitive and transparent: the VaR estimate is directly traceable to observed returns.

*Limitations:* Hostage to its lookback window: no crash in the sample means no crash in the estimate. The quantile is a step function of a finite sample, making it coarse and jumpy in the deep tail.

**Divergence Notes:**

Historical serves as the baseline. On genuinely normal data it converges with parametric and Monte Carlo. On real returns it diverges because the empirical tail captures fat tails and skew that the other two methods' normal assumption cannot see. The gap typically widens at 99% vs. 95% confidence, where tail shape matters most.

**Testing Notes:**

Reference values use a hand-built symmetric return series whose quantile positions are integers, so the expected quantile lands exactly on an order statistic with no interpolation. The independent cross-check draws a large normal sample and compares historical VaR and ES to the closed-form normal quantile and tail expectation via scipy, with a method-aware relative tolerance of 5% — sampling noise and the empirical quantile's coarseness make exact equality the wrong assertion.

---

## Methods: Parametric (variance-covariance)

**Overview:**

The parametric method assumes returns are normally distributed, estimates the mean and volatility from the sample, and reads VaR and ES off the closed-form normal tail. It is smooth, stable, and extrapolates beyond the worst observed return — properties that come at the cost of the normality assumption.

**Formulas:**

Let $z = \Phi^{-1}(\text{confidence})$, $\phi$ the standard-normal PDF, and $\alpha = 1 - \text{confidence}$.

  $VaR = (z \cdot \sigma - \mu) \cdot \sqrt{h} \cdot V$

  $ES = (\frac{\sigma \cdot \phi(z)}{\alpha} - \mu) \cdot \sqrt{h} \cdot V$

where $\mu$ and $\sigma$ are estimated from the sample, $h$ is the horizon in days, $V$ is the portfolio value.

**Design Choices:**

- Unbiased volatility (`ddof=1`). Dividing by $n - 1$ gives the correct estimator when inferring a population $\sigma$ from a sample. Reference tests pin this down by comparing against the exact `ddof=1` value.
- Mean estimated and subtracted, not assumed zero. Historical VaR already embeds drift through the empirical quantile. Treating $\mu$ identically in parametric ensures that any divergence between the two methods is attributable to the shape assumption, not a difference in drift handling. For 1-day horizons $\mu$ is usually negligible next to $z \cdot \sigma$, but the choice keeps the comparison honest.
- `scipy.stats.norm` supplies the `ppf` and `pdf`, `scipy` is permitted in the math core alongside `numpy`.
- $ES \geq VaR$ reinforcement. The closed-form normal tail guarantees this analytically: the truncated mean of a normal distribution always exceeds its truncation point.

**Strengths and Limitations:**

*Strengths:* Smooth and stable; it is not dependent on whether a single extreme day falls inside the window. Extrapolates into the tail beyond the worst observed return, which can make it more conservative than historical at very high confidence levels with short windows.

*Limitations:* Anchored to $z \cdot \sigma$ so it systematically understates deep-tail risk when returns exhibit excess kurtosis. Cannot capture skew: the normal is symmetric, and left-skewed real returns will cause the loss tail to be underestimated.

**Divergence Notes:**

On a genuinely normal sample, parametric and historical converge (cross-check asserts agreement within 5% on 200k–300k draws). On real returns they diverge due to fat tails and skew. The gap typically widens at 99% vs. 95% confidence. Monte Carlo, currently drawing from the same fitted normal, shares this blind spot and should converge to parametric — which both validates the parametric formula and sets up the natural extension of swapping the normal generator for a heavier-tailed distribution.

**Testing Notes:**

Reference tests compare computed VaR and ES against manually derived values using a known $\mu, \sigma,$ and confidence level. This pins the `ddof=1` estimator explicitly. The headline cross-check runs parametric against historical on a large normal sample (200k-300k draws) and asserts agreement within 5% relative tolerance.

---

## Methods: Monte Carlo

**Overview:**

The Monte Carlo method estimates $\mu$ and $\sigma$ exactly as the parametric method does, then simulates `n_sims` draws from $\mathcal{N}(\mu, \sigma)$ and reads VaR and ES off the simulated distribution. It reuses the same empirical quantile and tail-mean structure as the historical method. Its primary value is a cross-check on the parametric formula and as the method whose generating distribution is a free parameter.

**Formulas:**

Let $\{r_i\}_{i=1}^{\mathcal{N}}$ be draws from $\mathcal{N}(\mu, \sigma)$ where $\hat{\mu}$ and $\hat{\sigma}$ are estimated from the input returns.

  $VaR = -r_{\alpha}^{\text{sim}} \cdot \sqrt{h} \cdot V$

  $ES = - \frac{1}{|\mathcal{T}|} \sum_{r_{i} \in \mathcal{T}}r_{t} \cdot \sqrt{h} \cdot V$

where $r_{\alpha}^{\text{sim}}$ is the empirical `alpha`-quantile of the simulated draws, $\mathcal{T} = \{r_i : r_i \leq r_{\alpha}^{\text{sim}}\}$, $h$ is the horizon in days, and $V$ is the portfolio value.

**Design Choices:**

- Same estimator as parametric. Because Monte Carlo draws from the fitted normal, it must converge to the parametric closed form as $\mathcal{N} \rightarrow \infty$. That convergence is the headline test and validates both methods simultaneously.
- Reproducibility via `seed`. A keyword-only `seed` makes any run repeatable. `risk_report` passes the same seed to MC VaR and ES so both draw from one simulation; it ensures $ES \geq VaR$ holds exactly rather than probabilistically across two independent draws.
- `n_sims` default to 100k. Sufficient for a stable 99% tail while remaining fast. Validated as `>= 1` in the MC branch rather than in shared `validate_inputs`, following the same pattern as the parametric "$\geq 2$ returns" guard.
- Horizon via the shared $\sqrt{h}$ rule. MC simulates single-period returns and scales by $\sqrt{h}$, keeping all three methods comparable. Simulating multi-step compounded paths is the natural extension for a non-normal step distribution.

**Strengths and Limitations:**

*Strengths:* Validates the parametric formula by converging to it on a normal model. The generating distribution is a free parameter. Swapping the normal draw for a Student-$t$ or an empirical bootstrap would let MC capture fat tails while retaining the smooth, tail-extrapolating character that historical lacks.

*Limitations:* As currently built with a Gaussian generator, MC carries the same normality assumption as parametric and shares its blind spot to fat tails and skew. Simulation variance means results are not exact even at high `n_sims`, unlike the parametric closed form.

**Divergence Notes:**

On a normal model, MC converges to parametric (within ~3-4% at 100k-200k draws), confirming both methods. On real returns, MC diverges from historical for the same reason as parametric; they diverge at fat tails and the skew that the generator cannot reproduce. Replacing the generator with a non-normal distribution is likely the most promising lever for closing this gap.

**Testing Notes:**

The headline test asserts that MC VaR and ES agree with their parametric counterparts within ~3-4% relative tolerance at 100k-200k draws. Reproducibility is verified by running the same seed twice and asserting identical outputs. The `n_sims >= 1` guard is tested explicitly in the MC validation path.

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

## Closing the loop — ingestion, returns, and the end-to-end CLI

Real prices in, a stored, explained risk comparison out.

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

## The loop is closed

`var-model run` now goes from live tickers to a stored, explained risk
comparison; `var-model run --no-fetch` and `var-model history` work entirely off
the cache. Prices and returns are cached idempotently, results are persisted for
cross-run comparison, and the whole pipeline is tested without touching the
network. The pure `numpy`/`scipy` core remains free of any I/O.

---

## Phase 4 — validation against real portfolio data

The first live run validated the whole stack end to end and turned up one real
constraint.

**Free-tier constraint found in flight.** `outputsize=full` is now a premium
feature for `TIME_SERIES_DAILY`; the free tier returns an `Information` message
instead of data. The parser surfaced Alpha Vantage's own message verbatim (the
error handling working as designed), and the default was changed to `compact`
(~100 points), with `--full` left available for premium keys. *Consequence:* the
free-tier window is ~100 trading days, not the aspirational ~504.

**The run (AAPL/JPM/XOM/JNJ/PG, equal weight, ~99 returns, $1M):**

| Confidence | Historical VaR | Parametric VaR | Monte Carlo VaR |
|---|---|---|---|
| 95% | 12,475 | 11,557 | 11,565 |
| 99% | 14,670 | 16,422 | 16,551 |

Portfolio character over the window: annualized vol ~11.3% (daily σ 0.71%),
skewness −0.21 (mild left skew), excess kurtosis −0.28 (slightly *thin*-tailed),
Jarque-Bera p = 0.60 (normality not rejected).

**The math is verified.** Hand-computing the parametric formula and the empirical
quantile directly from the cached returns reproduces the stored numbers exactly,
confirming the fetch → returns → portfolio → VaR pipeline is wired correctly on
real data.

**The numbers make intuitive sense.** ES ≥ VaR throughout; VaR and ES are both
monotone in confidence; ES/VaR ≈ 1.26 at 95% (right at the normal value). The
three methods agree within ~11%, which is exactly what JB p = 0.60 predicts —
the headline equivalence (*methods agree ⇔ returns ≈ normal*) holds on live data,
not just synthetic.

**The instructive finding: a 95% ↔ 99% cross-over.**

- At **95%** historical is the *highest* (12,475 vs ~11,560): the empirical
  5th-percentile loss is lifted by the mild **left skew**, which the symmetric
  parametric/Monte Carlo methods cannot represent.
- At **99%** historical is the *lowest* (14,670 vs ~16,500): there the
  normal-based methods **extrapolate** the tail (`z = 2.33·σ`), while historical
  just reads the data — and with **thin tails** the empirical 1st-percentile is
  milder than the normal extrapolation.

This single portfolio shows both directions of divergence: historical tracks the
*actual* shape (skew, thin tails), the normal-based methods track an *assumed*
one. It is the project's thesis confirmed on real data.

**Caveat — small-sample tail.** With only ~99 observations the 99% historical
estimate interpolates between the single worst and second-worst days: coarse and
fragile. At 99% on free-tier data, historical is the *least* trustworthy of the
three; at 95% all three rest on solid footing. This is the documented
"historical is hostage to its window" limitation, made acute by the 100-point
cap. Two remedies without paying: the price cache **accumulates** across daily
runs (idempotent upserts grow the history past 100 points over time), and a
premium key unlocks `--full` for an immediate deep window.

---

## Project status: complete

All planned phases are done — scaffold, the three methods + ES, the divergence
analysis, SQL persistence, live ingestion + CLI, and validation on real data —
with a green five-pillar test suite throughout and the pure core kept free of
I/O. Natural extensions (not required by the original goals) are noted across
these design sections: a fat-tailed or bootstrap Monte Carlo generator, multi-day
compounded simulation paths, split/dividend-adjusted prices, and custom
portfolio weights.
