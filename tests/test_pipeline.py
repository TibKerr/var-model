"""Tests for the end-to-end pipeline and the CLI (no real network)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from var_model.cli import main
from var_model.data import (
    init_db,
    load_returns,
    make_engine,
    run_portfolio_analysis,
    save_prices,
)


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def _price_series(n: int = 60, base: float = 100.0, step: float = 0.5) -> pd.Series:
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series([base + i * step for i in range(n)], index=idx)


def _seed(session: Session, tickers: list[str]) -> None:
    save_prices(session, {t: _price_series(base=100.0 + 10 * i) for i, t in enumerate(tickers)})


# --- fake HTTP client (mirrors the fetch tests) ------------------------------

class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, tickers: list[str]) -> None:
        idx = pd.bdate_range("2023-01-02", periods=60)
        self._by_symbol = {
            t: {
                "Time Series (Daily)": {
                    str(d.date()): {"4. close": str(100.0 + 10 * i + j * 0.5)}
                    for j, d in enumerate(idx)
                }
            }
            for i, t in enumerate(tickers)
        }

    def get(self, url: str, params: dict[str, str], timeout: float) -> _FakeResponse:
        return _FakeResponse(self._by_symbol[params["symbol"]])


# --- pipeline ----------------------------------------------------------------

def test_pipeline_offline_persists_run_and_returns() -> None:
    tickers = ["AAPL", "JPM"]
    with _session() as session:
        _seed(session, tickers)
        run = run_portfolio_analysis(session, tickers, fetch=False, seed=0, label="offline")
        assert run.id is not None
        assert {r.method for r in run.results} == {"historical", "parametric", "monte_carlo"}
        # Computed returns were persisted too.
        assert not load_returns(session, tickers).empty


def test_pipeline_window_limits_observations() -> None:
    tickers = ["AAPL", "JPM"]
    with _session() as session:
        _seed(session, tickers)  # 60 prices -> 59 returns
        run = run_portfolio_analysis(session, tickers, fetch=False, window=30, seed=0)
        assert run.n_observations == 30


def test_pipeline_fetch_path_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("var_model.data.fetch.time.sleep", lambda s: None)
    tickers = ["AAPL", "JPM", "XOM"]
    client = _FakeClient(tickers)
    with _session() as session:
        run = run_portfolio_analysis(
            session, tickers, fetch=True, api_key="demo", client=client, seed=0
        )
        assert run.n_observations > 0
        assert len(run.results) == 3


def test_pipeline_offline_without_prices_raises() -> None:
    with _session() as session:
        with pytest.raises(RuntimeError, match="no cached prices"):
            run_portfolio_analysis(session, ["AAPL"], fetch=False)


# --- CLI ---------------------------------------------------------------------

def test_cli_run_offline_prints_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = make_engine(url)
    init_db(engine)
    with Session(engine) as session:
        _seed(session, ["AAPL", "JPM"])
    engine.dispose()

    main(["run", "--db", url, "--no-fetch", "--tickers", "AAPL", "JPM", "--seed", "0"])
    out = capsys.readouterr().out
    assert "Run #" in out
    assert "historical" in out and "parametric" in out and "monte_carlo" in out
    assert "diagnostics" in out


def test_cli_history_lists_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = make_engine(url)
    init_db(engine)
    with Session(engine) as session:
        _seed(session, ["AAPL", "JPM"])
    engine.dispose()

    args = ["run", "--db", url, "--no-fetch", "--tickers", "AAPL", "JPM", "--seed", "0"]
    main([*args, "--label", "first"])
    capsys.readouterr()  # clear
    main(["history", "--db", url])
    out = capsys.readouterr().out
    assert "first" in out


def test_cli_history_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    main(["history", "--db", url])
    assert "No runs stored yet." in capsys.readouterr().out
