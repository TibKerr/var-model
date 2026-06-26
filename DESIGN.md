# Design Notes

This document records the *why* behind `var-model` — the design decisions that
aren't obvious from the code, with particular emphasis (in later phases) on the
divergence analysis that compares the three VaR methods.

Each phase appends its own section as its logic lands.

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
