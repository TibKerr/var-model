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
