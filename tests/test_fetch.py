"""Tests for the Alpha Vantage fetch client (HTTP mocked, no real network)."""

from typing import Any

import pandas as pd
import pytest

from var_model.data.fetch import (
    RATE_LIMIT_SLEEP_SECONDS,
    fetch_daily_prices,
    fetch_portfolio_prices,
)


def _payload(closes: dict[str, float]) -> dict[str, Any]:
    """Build a minimal Alpha Vantage TIME_SERIES_DAILY payload."""
    return {
        "Meta Data": {"2. Symbol": "TEST"},
        "Time Series (Daily)": {
            day: {"4. close": str(close)} for day, close in closes.items()
        },
    }


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Stands in for ``requests``; records the params of each call."""

    def __init__(self, by_symbol: dict[str, dict[str, Any]]) -> None:
        self.by_symbol = by_symbol
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, params: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append(params)
        return _FakeResponse(self.by_symbol[params["symbol"]])


def test_parses_closes_sorted_ascending() -> None:
    # Deliberately out of order in the payload.
    client = _FakeClient({"AAPL": _payload({"2024-01-03": 102.0, "2024-01-02": 101.0})})
    series = fetch_daily_prices("AAPL", api_key="demo", client=client)
    assert list(series.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert series.iloc[0] == pytest.approx(101.0)
    assert series.iloc[1] == pytest.approx(102.0)
    assert series.name == "AAPL"
    assert client.calls[0]["function"] == "TIME_SERIES_DAILY"
    assert client.calls[0]["outputsize"] == "full"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"Note": "rate limit hit"}, "Note"),
        ({"Error Message": "invalid symbol"}, "Error Message"),
        ({"Information": "premium endpoint"}, "Information"),
        ({"Something Else": 1}, "Unexpected"),
    ],
)
def test_surfaces_api_error_responses(payload: dict[str, Any], match: str) -> None:
    client = _FakeClient({"X": payload})
    with pytest.raises(RuntimeError, match=match):
        fetch_daily_prices("X", api_key="demo", client=client)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        fetch_daily_prices("AAPL")


def test_env_api_key_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "env-key")
    client = _FakeClient({"AAPL": _payload({"2024-01-02": 10.0})})
    fetch_daily_prices("AAPL", client=client)
    assert client.calls[0]["apikey"] == "env-key"


def test_portfolio_fetch_returns_all_without_throttle() -> None:
    client = _FakeClient(
        {t: _payload({"2024-01-02": 10.0}) for t in ("AAPL", "JPM", "XOM")}
    )
    out = fetch_portfolio_prices(
        ["AAPL", "JPM", "XOM"], api_key="demo", client=client, throttle=False
    )
    assert set(out) == {"AAPL", "JPM", "XOM"}
    assert len(client.calls) == 3


def test_portfolio_fetch_throttles_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("var_model.data.fetch.time.sleep", lambda s: sleeps.append(s))
    client = _FakeClient({t: _payload({"2024-01-02": 1.0}) for t in ("A", "B", "C")})
    fetch_portfolio_prices(["A", "B", "C"], api_key="demo", client=client)
    # n - 1 sleeps, each the rate-limit interval.
    assert sleeps == [RATE_LIMIT_SLEEP_SECONDS, RATE_LIMIT_SLEEP_SECONDS]
