"""Alpha Vantage ingestion — the only network code in the project.

Calls the free ``TIME_SERIES_DAILY`` endpoint directly with ``requests`` and
returns daily closing prices as pandas Series. The API key is read from the
``ALPHAVANTAGE_API_KEY`` environment variable (or passed explicitly); it is never
hard-coded. The free tier allows ~5 requests/minute, so multi-ticker fetches
throttle between calls.

This module lives in the I/O layer; the math core never imports it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any, Protocol

import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"

# Free tier is ~5 requests/minute; 12s between calls stays comfortably under it.
RATE_LIMIT_SLEEP_SECONDS = 12.0


class _Getter(Protocol):
    """Minimal interface for the HTTP client (``requests`` or a test double)."""

    def get(self, url: str, params: dict[str, str], timeout: float) -> Any: ...


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not set; put it in .env (see .env.example) "
            "or pass api_key explicitly"
        )
    return key


def _parse_daily(ticker: str, payload: dict[str, Any]) -> pd.Series:
    """Turn an Alpha Vantage daily payload into a sorted close-price Series."""
    series_block = payload.get("Time Series (Daily)")
    if series_block is None:
        # Surface Alpha Vantage's own error / rate-limit messages verbatim.
        for field in ("Error Message", "Note", "Information"):
            if field in payload:
                raise RuntimeError(
                    f"Alpha Vantage returned '{field}' for {ticker}: {payload[field]}"
                )
        raise RuntimeError(
            f"Unexpected Alpha Vantage response for {ticker}: keys={list(payload)}"
        )
    closes = {
        pd.Timestamp(day): float(fields["4. close"])
        for day, fields in series_block.items()
    }
    return pd.Series(closes, name=ticker, dtype=float).sort_index()


def fetch_daily_prices(
    ticker: str,
    *,
    api_key: str | None = None,
    outputsize: str = "compact",
    client: _Getter | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch the daily closing-price series for one ticker.

    ``outputsize`` is ``"compact"`` (last 100 points; the default and the only
    free-tier option) or ``"full"`` (full history, a premium feature). ``client``
    defaults to ``requests``; tests inject a double. Raises ``RuntimeError`` with
    Alpha Vantage's message on an error or rate-limit response.
    """
    key = _resolve_key(api_key)
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": key,
    }
    if client is None:
        response = requests.get(BASE_URL, params=params, timeout=timeout)
    else:
        response = client.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return _parse_daily(ticker, response.json())


def fetch_portfolio_prices(
    tickers: Iterable[str],
    *,
    api_key: str | None = None,
    outputsize: str = "compact",
    client: _Getter | None = None,
    throttle: bool = True,
    sleep_seconds: float = RATE_LIMIT_SLEEP_SECONDS,
) -> dict[str, pd.Series]:
    """Fetch closes for several tickers, throttling between calls for the free tier.

    Returns a dict mapping ticker -> close Series, ready for ``save_prices``.
    """
    key = _resolve_key(api_key)
    prices: dict[str, pd.Series] = {}
    for index, ticker in enumerate(tickers):
        if throttle and index > 0:
            time.sleep(sleep_seconds)
        prices[ticker] = fetch_daily_prices(
            ticker, api_key=key, outputsize=outputsize, client=client
        )
    return prices
