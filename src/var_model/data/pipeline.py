"""End-to-end orchestration: fetch -> compute -> persist.

This ties the data layer and the math core into one pipeline so the CLI can stay
thin. It is the only place that imports across both layers; the math core remains
unaware of fetching and storage.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from var_model.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON,
    DEFAULT_N_SIMS,
    DEFAULT_VALUE,
    DEFAULT_WINDOW,
)
from var_model.data.database import (
    compute_and_save,
    load_prices,
    save_prices,
    save_returns,
)
from var_model.data.fetch import _Getter, fetch_portfolio_prices
from var_model.data.returns import log_returns, portfolio_returns
from var_model.data.schema import Run


def run_portfolio_analysis(
    session: Session,
    tickers: Sequence[str],
    *,
    fetch: bool = True,
    api_key: str | None = None,
    client: _Getter | None = None,
    outputsize: str = "compact",
    confidence: float = DEFAULT_CONFIDENCE,
    horizon: int = DEFAULT_HORIZON,
    value: float = DEFAULT_VALUE,
    window: int | None = DEFAULT_WINDOW,
    weights: Sequence[float] | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int | None = None,
    label: str | None = None,
) -> Run:
    """Run the full pipeline and persist the result.

    Steps: optionally fetch prices from Alpha Vantage and cache them; load the
    cached prices; compute and store per-asset log returns; aggregate to the
    portfolio series; restrict to the trailing ``window``; then compute the
    divergence report and save it as a Run. Returns the saved Run.
    """
    if fetch:
        save_prices(
            session,
            fetch_portfolio_prices(
                tickers, api_key=api_key, client=client, outputsize=outputsize
            ),
        )

    prices = load_prices(session, list(tickers))
    if prices.empty:
        raise RuntimeError(
            "no cached prices for the requested tickers; run with fetching enabled"
        )

    asset_returns = log_returns(prices)
    save_returns(session, asset_returns)

    portfolio = portfolio_returns(asset_returns, weights)
    if window is not None and 0 < window < portfolio.shape[0]:
        portfolio = portfolio[-window:]

    return compute_and_save(
        session,
        portfolio,
        confidence=confidence,
        horizon=horizon,
        value=value,
        n_sims=n_sims,
        seed=seed,
        label=label,
    )
