"""Price -> return transformation and portfolio aggregation.

Data-preparation layer (uses pandas), kept out of the pure ``numpy``/``scipy``
math core. It turns a wide price frame into per-asset log returns and aggregates
those into the single portfolio return series the risk methods consume.

Return convention (set in agreement with the project): **log returns**,
``r = ln(P_t / P_{t-1})``. The portfolio return is the weighted sum of asset log
returns, ``r_p = sum_i w_i r_i``. This is the standard approximation (exact in
the continuous-compounding limit; the error is negligible at daily horizons) and
it keeps the portfolio a linear combination of the asset returns, which matches
the variance-covariance method's mental model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset daily log returns, with the first (undefined) row dropped.

    ``prices`` is a date-indexed frame with one column per ticker. Raises
    ``ValueError`` on too-few rows or non-positive prices (log undefined).
    """
    if prices.shape[0] < 2:
        raise ValueError("prices must have at least 2 rows to compute returns")
    if (prices <= 0).to_numpy().any():
        raise ValueError("prices must be positive to take log returns")
    values = prices.to_numpy(dtype=np.float64)
    ratios = values[1:] / values[:-1]
    return pd.DataFrame(np.log(ratios), index=prices.index[1:], columns=prices.columns)


def portfolio_returns(
    returns: pd.DataFrame, weights: Sequence[float] | None = None
) -> NDArray[np.float64]:
    """Aggregate per-asset log returns into one portfolio return series.

    ``returns`` is the output of :func:`log_returns`. ``weights`` defaults to
    equal weight (``1/N`` per asset); if given it must have one weight per column
    and sum to 1. Returns a 1-D array ready for the risk methods.
    """
    if returns.empty:
        raise ValueError("returns must be non-empty")
    n_assets = returns.shape[1]
    if weights is None:
        w = np.full(n_assets, 1.0 / n_assets)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n_assets:
            raise ValueError(
                f"weights has {w.shape[0]} entries but there are {n_assets} assets"
            )
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {float(w.sum())}")
    return np.asarray(returns.to_numpy(dtype=np.float64) @ w, dtype=np.float64)
