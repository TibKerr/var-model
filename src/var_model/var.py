"""Primary Value-at-Risk computation and shared validation for var_model.

This is the pure numerical core: it imports only ``numpy`` (and later ``scipy``)
— no network, no database. Every method takes the same validated ``returns``
array and the same parameters, so the three approaches (historical, parametric,
Monte Carlo) can be compared on *identical* data. That comparison is the
project's headline deliverable.

Conventions
-----------
- ``confidence`` is the VaR confidence level in (0, 1); the tail probability is
  ``alpha = 1 - confidence`` (e.g. ``confidence=0.95`` → the 5th percentile).
- VaR is returned as a **positive loss** in the same units as ``value``
  (default ``value=1.0``, so the result is a loss *fraction* of the portfolio).
- Multi-day horizons use the square-root-of-time rule.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Method = Literal["historical", "parametric", "monte_carlo"]

_METHODS: tuple[Method, ...] = ("historical", "parametric", "monte_carlo")


def validate_inputs(
    returns: ArrayLike,
    confidence: float,
    horizon: int,
    value: float,
    method: Method,
) -> NDArray[np.float64]:
    """Validate the shared inputs and return ``returns`` as a 1-D float array.

    Raises ``ValueError`` naming the offending argument on any out-of-domain
    input. Returning the cleaned array lets callers validate and normalize in a
    single step.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("returns must be a non-empty 1-D array of returns")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite values (no NaN/inf)")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return arr


def _quantile(returns: NDArray[np.float64], confidence: float) -> float:
    """Empirical lower-tail return at the given confidence.

    For confidence ``c`` the tail probability is ``alpha = 1 - c``; this returns
    the ``alpha``-quantile of the return distribution (a return, typically
    negative) using linear interpolation between order statistics. Shared by the
    historical VaR and the historical Expected Shortfall.
    """
    alpha = 1.0 - confidence
    return float(np.quantile(returns, alpha, method="linear"))


def value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
    method: Method = "historical",
    horizon: int = 1,
    value: float = 1.0,
) -> float:
    """Value-at-Risk: the loss not exceeded with probability ``confidence``.

    Returned as a positive loss in the same units as ``value`` (default 1.0, so
    the result is a loss fraction of the portfolio).

    Historical VaR sorts the observed returns and reads off the empirical
    ``(1 - confidence)`` quantile — no distributional assumption. The parametric
    and Monte Carlo methods land in later milestones.
    """
    arr = validate_inputs(returns, confidence, horizon, value, method)
    if method == "historical":
        q = _quantile(arr, confidence)
        return float(-q * np.sqrt(horizon) * value)
    raise NotImplementedError(f"{method!r} VaR is not implemented yet")
