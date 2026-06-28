"""Default analysis parameters.

Centralized so the CLI and any scripts share one source of truth for the
portfolio composition and the standard run settings. These are defaults only;
every one is overridable per run.
"""

from __future__ import annotations

# Diversified 5-name portfolio (tech, financials, energy, healthcare, staples).
DEFAULT_TICKERS: tuple[str, ...] = ("AAPL", "JPM", "XOM", "JNJ", "PG")

# Trailing observations used for the risk estimate (~2 years of trading days).
DEFAULT_WINDOW = 504

DEFAULT_CONFIDENCE = 0.95
DEFAULT_HORIZON = 1
DEFAULT_VALUE = 1_000_000.0
DEFAULT_N_SIMS = 100_000
