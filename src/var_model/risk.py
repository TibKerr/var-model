"""Secondary risk metrics (Expected Shortfall) and the bundle helper.

Like ``var.py`` this is part of the pure numerical core — ``numpy`` only, no
network or database. Expected Shortfall (a.k.a. CVaR) answers the question VaR
leaves open: *given* a loss past the VaR threshold, how bad is it on average?

``risk_report`` bundles VaR and ES across the implemented methods, computed on
the same returns, so a caller gets the full picture in one call. It grows new
keys as the parametric and Monte Carlo methods land.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from var_model.var import Method, _normal_params, _quantile, validate_inputs, value_at_risk


def expected_shortfall(
    returns: ArrayLike,
    confidence: float = 0.95,
    method: Method = "historical",
    horizon: int = 1,
    value: float = 1.0,
) -> float:
    """Expected Shortfall: the mean loss in the worst ``(1 - confidence)`` tail.

    Returned as a positive loss in the same units as ``value``. By construction
    ``ES >= VaR`` (the average of the tail is at least as extreme as its
    threshold).

    - **historical** averages every observed return at or below the empirical
      ``(1 - confidence)`` quantile — no distributional assumption.
    - **parametric** uses the closed-form normal tail expectation
      ``ES = sigma·phi(z)/(1 - confidence) - mu`` with ``z = Phi^{-1}(c)``.
    """
    arr = validate_inputs(returns, confidence, horizon, value, method)
    if method == "historical":
        q = _quantile(arr, confidence)
        tail = arr[arr <= q]  # q is an order statistic, so the tail is non-empty
        return float(-tail.mean() * np.sqrt(horizon) * value)
    if method == "parametric":
        mu, sigma = _normal_params(arr)
        z = float(norm.ppf(confidence))
        es = sigma * float(norm.pdf(z)) / (1.0 - confidence) - mu
        return float(es * np.sqrt(horizon) * value)
    raise NotImplementedError(f"{method!r} Expected Shortfall is not implemented yet")


def risk_report(
    returns: ArrayLike,
    confidence: float = 0.95,
    horizon: int = 1,
    value: float = 1.0,
) -> dict[str, float]:
    """Bundle VaR and ES across the implemented methods on the same returns."""
    return {
        "var_historical": value_at_risk(returns, confidence, "historical", horizon, value),
        "es_historical": expected_shortfall(returns, confidence, "historical", horizon, value),
        "var_parametric": value_at_risk(returns, confidence, "parametric", horizon, value),
        "es_parametric": expected_shortfall(returns, confidence, "parametric", horizon, value),
    }
