"""Comparative (divergence) analysis across the three VaR/ES methods.

Pure numerical core — ``numpy``/``scipy`` only, no network or database. This is
the project's headline deliverable: it runs all three methods on one return
series, quantifies how far apart they land (the *spread*), and reports the
distribution-shape diagnostics that explain *why* they diverge.

The causal chain the diagnostics expose: the parametric and Monte Carlo methods
assume normality, so they are pinned to ``z·sigma``. When real returns are
**skewed** or **fat-tailed** (excess kurtosis > 0) the historical method — which
reads an actual empirical quantile — pulls away from them in the tail. A low
Jarque-Bera p-value is the statistical signal that this divergence is expected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import jarque_bera, kurtosis, skew

from var_model.risk import risk_report

# Conventional significance level for reading the Jarque-Bera test: a p-value
# below this rejects normality, flagging the regime where parametric/MC
# understate the tail historical captures.
NORMALITY_ALPHA = 0.05


def distribution_diagnostics(returns: ArrayLike) -> dict[str, float]:
    """Shape statistics that explain (or rule out) method divergence.

    Returns the sample size, mean, std, skewness, excess kurtosis (Fisher; a
    normal distribution scores 0), and the Jarque-Bera normality test statistic
    and p-value. A p-value below ``NORMALITY_ALPHA`` flags non-normal returns —
    the regime where parametric and Monte Carlo understate the tail.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError("returns must contain at least 2 observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must contain only finite values (no NaN/inf)")
    std = float(arr.std(ddof=1))
    if std == 0.0:
        # Degenerate: no variance means no shape and no divergence to explain.
        return {
            "n_observations": float(arr.size),
            "mean": float(arr.mean()),
            "std": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "jarque_bera": 0.0,
            "jarque_bera_pvalue": 1.0,
        }
    jb = jarque_bera(arr)
    return {
        "n_observations": float(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "skewness": float(skew(arr)),
        "excess_kurtosis": float(kurtosis(arr)),
        "jarque_bera": float(jb.statistic),
        "jarque_bera_pvalue": float(jb.pvalue),
    }


def _spread(values: tuple[float, ...]) -> tuple[float, float]:
    """Absolute and relative spread (max - min) of a set of estimates.

    The relative spread is normalized by the largest (most conservative)
    estimate, so it reads as a fraction: 0 when the methods agree, growing as
    they diverge.
    """
    lo, hi = min(values), max(values)
    spread = hi - lo
    relative = spread / abs(hi) if hi != 0.0 else 0.0
    return spread, relative


def divergence_report(
    returns: ArrayLike,
    confidence: float = 0.95,
    horizon: int = 1,
    value: float = 1.0,
    *,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Full comparison: the six risk numbers, their spread, and the diagnostics.

    Combines ``risk_report`` (VaR + ES across all three methods on identical
    data), the absolute and relative spread between methods for each metric, and
    the distribution-shape diagnostics that explain any divergence — everything a
    caller needs to both report the numbers and justify them.
    """
    report = risk_report(returns, confidence, horizon, value, n_sims=n_sims, seed=seed)
    var_spread, var_rel = _spread(
        (report["var_historical"], report["var_parametric"], report["var_monte_carlo"])
    )
    es_spread, es_rel = _spread(
        (report["es_historical"], report["es_parametric"], report["es_monte_carlo"])
    )
    return {
        **report,
        "var_spread": var_spread,
        "var_spread_relative": var_rel,
        "es_spread": es_spread,
        "es_spread_relative": es_rel,
        **distribution_diagnostics(returns),
    }
