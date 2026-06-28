"""Engine/session handling and persistence of risk results to SQL.

The math core produces plain result dicts; this module writes them to the
database and reads them back. ``save_divergence_report`` persists a dict that was
already computed (keeping compute and I/O separable), while ``compute_and_save``
is the end-to-end convenience that runs the analysis and stores it in one call.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd
from numpy.typing import ArrayLike
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from var_model.data.schema import Base, MethodResult, Price, Run
from var_model.divergence import divergence_report

DEFAULT_DB_URL = "sqlite:///var_model.db"

_METHODS = ("historical", "parametric", "monte_carlo")


def make_engine(url: str | None = None) -> Engine:
    """Create an Engine from an explicit URL, the ``VAR_MODEL_DB_URL`` env var,
    or the default local SQLite file."""
    return create_engine(url or os.environ.get("VAR_MODEL_DB_URL", DEFAULT_DB_URL))


def init_db(engine: Engine) -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(engine)


def save_divergence_report(
    session: Session,
    report: dict[str, float],
    *,
    confidence: float,
    horizon: int = 1,
    value: float = 1.0,
    n_sims: int = 100_000,
    seed: int | None = None,
    label: str | None = None,
) -> Run:
    """Persist an already-computed ``divergence_report`` dict as one Run with
    three child MethodResult rows. Commits and returns the populated Run."""
    run = Run(
        confidence=confidence,
        horizon=horizon,
        value=value,
        n_sims=n_sims,
        seed=seed,
        label=label,
        n_observations=int(report["n_observations"]),
        mean=report["mean"],
        std=report["std"],
        skewness=report["skewness"],
        excess_kurtosis=report["excess_kurtosis"],
        jarque_bera=report["jarque_bera"],
        jarque_bera_pvalue=report["jarque_bera_pvalue"],
        var_spread=report["var_spread"],
        var_spread_relative=report["var_spread_relative"],
        es_spread=report["es_spread"],
        es_spread_relative=report["es_spread_relative"],
    )
    run.results = [
        MethodResult(method=m, var=report[f"var_{m}"], es=report[f"es_{m}"])
        for m in _METHODS
    ]
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def compute_and_save(
    session: Session,
    returns: ArrayLike,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    value: float = 1.0,
    n_sims: int = 100_000,
    seed: int | None = None,
    label: str | None = None,
) -> Run:
    """Run the divergence analysis on ``returns`` and persist it in one call."""
    report = divergence_report(
        returns, confidence, horizon, value, n_sims=n_sims, seed=seed
    )
    return save_divergence_report(
        session,
        report,
        confidence=confidence,
        horizon=horizon,
        value=value,
        n_sims=n_sims,
        seed=seed,
        label=label,
    )


def load_runs(session: Session) -> list[Run]:
    """Return all persisted runs, oldest first."""
    return list(session.scalars(select(Run).order_by(Run.created_at, Run.id)))


def save_prices(session: Session, prices: Mapping[str, pd.Series]) -> int:
    """Upsert daily closing prices, keyed by ticker.

    ``prices`` maps each ticker to a Series of closes indexed by date. Existing
    (ticker, date) rows are updated in place rather than duplicated, so ingestion
    is idempotent. Returns the number of rows inserted or updated.
    """
    written = 0
    for ticker, series in prices.items():
        existing = {
            row.date: row
            for row in session.scalars(select(Price).where(Price.ticker == ticker))
        }
        for raw_date, close in series.items():
            day = pd.Timestamp(str(raw_date)).date()
            row = existing.get(day)
            if row is None:
                session.add(Price(ticker=ticker, date=day, close=float(close)))
            else:
                row.close = float(close)
            written += 1
    session.commit()
    return written


def load_prices(session: Session, tickers: list[str]) -> pd.DataFrame:
    """Load stored closes for ``tickers`` as a wide DataFrame.

    Index is the date, one column per ticker (ordered as requested), sorted
    ascending. Rows where any requested ticker is missing are dropped so the
    frame is aligned and ready for return computation.
    """
    stmt = select(Price).where(Price.ticker.in_(tickers)).order_by(Price.date)
    rows = session.scalars(stmt).all()
    if not rows:
        return pd.DataFrame(columns=tickers)
    frame = pd.DataFrame(
        {
            "date": [r.date for r in rows],
            "ticker": [r.ticker for r in rows],
            "close": [r.close for r in rows],
        }
    )
    wide = frame.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(columns=tickers).sort_index().dropna()


__all__ = [
    "DEFAULT_DB_URL",
    "compute_and_save",
    "init_db",
    "load_prices",
    "load_runs",
    "make_engine",
    "save_divergence_report",
    "save_prices",
]
