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
- **Single dependency set.** `numpy`, `scipy`, `pandas`, `sqlalchemy`, and
  `yfinance` are all required runtime dependencies — one `uv sync` installs
  everything. *However,* the math modules (`var.py`, `risk.py`) are kept to
  `numpy`/`scipy` only at the code level so the risk core stays pure and its
  tests never depend on the network (`yfinance`) or a database (`sqlalchemy`).
  The separation is by module responsibility, not by package extras.
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
