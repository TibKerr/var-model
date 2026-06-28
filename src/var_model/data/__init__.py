"""I/O layer for var_model: SQL persistence of risk/divergence results.

This package owns everything that touches the database. The math core
(``var.py``, ``risk.py``, ``divergence.py``) stays pure and never imports from
here; persistence consumes the plain result dicts the core produces.
"""

from var_model.data.database import (
    compute_and_save,
    init_db,
    load_prices,
    load_returns,
    load_runs,
    make_engine,
    save_divergence_report,
    save_prices,
    save_returns,
)
from var_model.data.fetch import fetch_daily_prices, fetch_portfolio_prices
from var_model.data.returns import log_returns, portfolio_returns
from var_model.data.schema import Base, MethodResult, Price, Return, Run

__all__ = [
    "Base",
    "MethodResult",
    "Price",
    "Return",
    "Run",
    "compute_and_save",
    "fetch_daily_prices",
    "fetch_portfolio_prices",
    "init_db",
    "load_prices",
    "load_returns",
    "load_runs",
    "log_returns",
    "make_engine",
    "portfolio_returns",
    "save_divergence_report",
    "save_prices",
    "save_returns",
]
