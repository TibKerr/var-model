# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`var-model` computes Value-at-Risk and Expected Shortfall for a small equity
portfolio three independent ways — **historical**, **variance-covariance
(parametric)**, and **Monte Carlo**. The comparison of those methods and the
explanation of *why they diverge* is the project's headline deliverable, not an
afterthought. Price data comes from the Alpha Vantage REST API (called directly
with `requests`); raw prices and computed returns are persisted to SQL via
SQLAlchemy, and risk results are written back to the same database for cross-run
comparison.

The Alpha Vantage key is read from the `ALPHAVANTAGE_API_KEY` environment
variable (loaded from a git-ignored `.env` via `python-dotenv` in dev) — never
hard-coded or committed. The free tier is rate-limited (~5 req/min, ~25/day), so
the data layer throttles and caches to the DB.

## Commands

This project is managed with `uv`. Prefix everything with `uv run` to execute
inside the project venv.

```
uv sync                       # create .venv, install package + dev tools, refresh uv.lock
uv run var-model --help       # run the CLI
uv run ruff check .           # lint + import-sort (matches CI)
uv run mypy                   # type check src + tests (matches CI)
uv run pytest -q              # full test suite (matches CI)
uv run pytest tests/test_var.py -k historical   # single file / single test by keyword
```

The three quality commands (`ruff`, `mypy`, `pytest`) are exactly what
`.github/workflows/ci.yml` runs — a green local run should mean a green
pipeline. Run all three before committing.

> If `uv run pytest` fails to spawn on this Windows machine ("Application Control
> policy has blocked this file"), use `uv run python -m pytest -q` instead — it
> invokes the same test run without the blocked console-script shim.

## Architecture

Modern `src` layout: importable code lives under `src/var_model/`, tests at the
top level, all config in `pyproject.toml`. Build backend is **hatchling** (not
uv's default `uv_build`).

The design deliberately separates a **pure math core** from an **I/O layer**:

- **Math core** (`var.py`, `risk.py`, `divergence.py`) imports only
  `numpy`/`scipy`. It takes arrays/numbers in and returns numbers out — no
  network, no database. This is what makes the math testable against synthetic
  distributions with no external dependencies. *Keep it that way:* do not import
  `requests` or `sqlalchemy` into these modules even though they are installed.
- **I/O + data-prep layer** (`data/`) owns everything that touches the network,
  disk, or pandas: SQLAlchemy models/session (`schema.py`, `database.py`), Alpha
  Vantage ingestion (`fetch.py`, `requests`), price→return/portfolio computation
  (`returns.py`, pandas), and the cross-layer orchestration (`pipeline.py`).
  Note: this `data/` package is the *data access/prep layer* (code), distinct
  from any folder of market data.
- **CLI** (`cli.py`) stays thin — it parses arguments, calls `pipeline`, and
  formats output. No business logic. Subcommands: `run` (fetch→compute→persist→
  print) and `history`.

All dependencies are a single required runtime set (numpy, scipy, pandas,
sqlalchemy, requests, python-dotenv); the core/IO separation is enforced by module
responsibility at the code level, not by packaging extras.

> Phase status (see `DESIGN.md` for the rationale behind each): **all phases
> complete** — scaffold; the three VaR methods + ES (`var.py`, `risk.py`); the
> divergence/comparison layer (`divergence.py`); SQL persistence; Alpha Vantage
> ingestion, returns/portfolio computation, and the end-to-end CLI (`data/`,
> `cli.py`). Full five-pillar test suite throughout.

## Working conventions (important — this project has strict process rules)

- **Phased development.** Work proceeds in discrete phases. Do not start the next
  phase until the current one is verified by the user.
- **Granular, single-purpose commits.** One logical change per commit. Never
  bundle unrelated changes (a schema change and a math function are two commits).
  Use Conventional Commits: `feat:`, `test:`, `fix:`, `chore:`, `docs:`,
  `refactor:`.
- **Tests accompany every math module.** Each VaR/ES module ships a matching
  `pytest` file built on the five pillars: (1) reference value vs. a known
  closed-form/published figure, (2) an independent cross-check (e.g. parametric
  vs. historical/Monte-Carlo on the same data), (3) invariants (VaR monotone in
  confidence; ES ≥ VaR), (4) edge cases (zero variance → zero VaR), (5)
  validation (bad inputs raise `ValueError` naming the offending argument).
- **Cross-method tests assert convergence, not equality.** Historical VaR is not
  always subadditive; test that methods converge on a known distribution with
  method-aware tolerances rather than asserting exact equality.
- **Document the "why" in `DESIGN.md`.** Append a section as each logic block
  lands, especially for the divergence analysis.

## Conventions for the math API

Primary functions take a small set of validated numeric inputs plus a
`method` discriminator (`"parametric" | "historical" | "monte_carlo"`).
Validation lives in a shared `validate_inputs`; out-of-domain arguments raise
`ValueError` with a message naming the offending argument. A bundle helper
(`risk_report`) returns several related outputs (VaR + ES across methods) in one
call.
