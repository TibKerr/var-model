"""Tests for the divergence analysis layer.

The headline assertions: on normal returns the three methods agree and normality
is not rejected; on fat-tailed returns the methods diverge in the tail and
Jarque-Bera rejects normality. Plus the usual reference / cross-check /
invariant / edge / validation pillars.
"""

import numpy as np
import pytest
from scipy.stats import t as student_t

from var_model import distribution_diagnostics, divergence_report, risk_report


def _normal(seed: int = 0, size: int = 5_000, sigma: float = 0.02) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, size)


def _fat_tailed(seed: int = 0, size: int = 5_000, sigma: float = 0.02) -> np.ndarray:
    raw = student_t.rvs(df=3, size=size, random_state=np.random.default_rng(seed))
    return raw / raw.std() * sigma


# --- Pillar 1: reference values (known distribution shapes) -------------------

def test_diagnostics_on_normal_are_near_zero_shape() -> None:
    d = distribution_diagnostics(_normal(size=20_000))
    assert d["skewness"] == pytest.approx(0.0, abs=0.1)
    assert d["excess_kurtosis"] == pytest.approx(0.0, abs=0.15)
    assert d["jarque_bera_pvalue"] > 0.05  # do not reject normality


def test_diagnostics_on_fat_tails_flag_kurtosis() -> None:
    d = distribution_diagnostics(_fat_tailed(size=20_000))
    assert d["excess_kurtosis"] > 1.0  # Student-t df=3 is heavily fat-tailed
    assert d["jarque_bera_pvalue"] < 0.01  # reject normality


# --- Pillar 2: independent cross-check (consistency with the pieces) ----------

def test_report_embeds_risk_report_values() -> None:
    returns = _normal()
    div = divergence_report(returns, confidence=0.99, seed=0)
    rep = risk_report(returns, confidence=0.99, seed=0)
    for key, val in rep.items():
        assert div[key] == pytest.approx(val)


def test_report_embeds_diagnostics() -> None:
    returns = _fat_tailed()
    div = divergence_report(returns, confidence=0.95, seed=0)
    diag = distribution_diagnostics(returns)
    for key, val in diag.items():
        assert div[key] == pytest.approx(val)


def test_spread_matches_manual_max_minus_min() -> None:
    returns = _fat_tailed()
    div = divergence_report(returns, confidence=0.99, seed=0)
    vars_ = [div["var_historical"], div["var_parametric"], div["var_monte_carlo"]]
    assert div["var_spread"] == pytest.approx(max(vars_) - min(vars_))
    assert div["var_spread_relative"] == pytest.approx(
        (max(vars_) - min(vars_)) / abs(max(vars_))
    )


# --- Pillar 3: invariants (the divergence thesis) ----------------------------

def test_fat_tails_diverge_more_than_normal() -> None:
    # The whole point: methods spread further apart on fat-tailed returns.
    normal = divergence_report(_normal(size=10_000), confidence=0.99, seed=0)
    fat = divergence_report(_fat_tailed(size=10_000), confidence=0.99, seed=0)
    assert fat["var_spread_relative"] > normal["var_spread_relative"]
    assert fat["es_spread_relative"] > normal["es_spread_relative"]


def test_spreads_are_non_negative() -> None:
    div = divergence_report(_normal(), confidence=0.95, seed=0)
    assert div["var_spread"] >= 0.0
    assert div["es_spread"] >= 0.0


def test_historical_exceeds_normal_methods_in_fat_tail() -> None:
    # At 99% on fat tails, the empirical quantile is more extreme than z*sigma.
    div = divergence_report(_fat_tailed(size=20_000), confidence=0.99, seed=0)
    assert div["var_historical"] > div["var_parametric"]
    assert div["es_historical"] > div["es_parametric"]


# --- Pillar 4: edge cases -----------------------------------------------------

def test_zero_variance_has_zero_spread_and_neutral_diagnostics() -> None:
    div = divergence_report([0.0] * 100, confidence=0.95, seed=0)
    assert div["var_spread"] == pytest.approx(0.0)
    assert div["es_spread"] == pytest.approx(0.0)
    assert div["std"] == pytest.approx(0.0)
    assert div["skewness"] == pytest.approx(0.0)
    assert div["excess_kurtosis"] == pytest.approx(0.0)
    assert div["jarque_bera_pvalue"] == pytest.approx(1.0)


# --- Pillar 5: validation -----------------------------------------------------

def test_diagnostics_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        distribution_diagnostics([0.01])


def test_diagnostics_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        distribution_diagnostics([0.01, np.inf, -0.02])


@pytest.mark.parametrize("kwargs", [{"confidence": 1.0}, {"horizon": 0}, {"value": 0.0}])
def test_report_rejects_bad_scalars(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        divergence_report(_normal(size=100), **kwargs)  # type: ignore[arg-type]
