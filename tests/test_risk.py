"""Tests for historical Expected Shortfall and the risk_report bundle.

Mirrors the five pillars, with ES's defining invariant (ES >= VaR) front and
center, plus the closed-form normal-tail expectation as the cross-check.
"""

import numpy as np
import pytest
from scipy.stats import norm

from var_model import expected_shortfall, risk_report, value_at_risk

SYMMETRIC = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


# --- Pillar 1: reference value ------------------------------------------------

def test_reference_value_tail_mean() -> None:
    # confidence 0.90 -> alpha 0.10 -> quantile -0.04. Tail = {-0.05, -0.04},
    # mean -0.045 -> ES = +0.045.
    assert expected_shortfall(SYMMETRIC, confidence=0.90) == pytest.approx(0.045)


# --- Pillar 2: independent cross-check ----------------------------------------

def test_matches_normal_tail_expectation() -> None:
    rng = np.random.default_rng(1)
    sigma = 0.02
    sample = rng.normal(loc=0.0, scale=sigma, size=300_000)
    alpha = 1.0 - 0.99
    # Closed form: ES = sigma * phi(Phi^{-1}(alpha)) / alpha for a zero-mean normal.
    theoretical = sigma * norm.pdf(norm.ppf(alpha)) / alpha
    assert expected_shortfall(sample, confidence=0.99) == pytest.approx(
        theoretical, rel=0.05
    )


# --- Pillar 3: invariants -----------------------------------------------------

def test_es_at_least_var() -> None:
    rng = np.random.default_rng(2)
    sample = rng.normal(0.0, 0.02, 50_000)
    for c in (0.90, 0.95, 0.99):
        assert expected_shortfall(sample, confidence=c) >= value_at_risk(
            sample, confidence=c
        )


def test_es_monotone_in_confidence() -> None:
    assert expected_shortfall(SYMMETRIC, confidence=0.99) >= expected_shortfall(
        SYMMETRIC, confidence=0.90
    )


def test_horizon_and_value_scaling() -> None:
    base = expected_shortfall(SYMMETRIC, confidence=0.90)
    scaled = expected_shortfall(SYMMETRIC, confidence=0.90, horizon=4, value=1_000_000)
    assert scaled == pytest.approx(base * 2.0 * 1_000_000)


# --- Pillar 4: edge cases -----------------------------------------------------

def test_zero_returns_give_zero_es() -> None:
    assert expected_shortfall([0.0] * 50, confidence=0.95) == pytest.approx(0.0)


# --- Pillar 5: validation -----------------------------------------------------

@pytest.mark.parametrize("kwargs", [{"confidence": 1.0}, {"horizon": 0}, {"value": -1.0}])
def test_validation_rejects_bad_scalars(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        expected_shortfall(SYMMETRIC, **kwargs)  # type: ignore[arg-type]


# --- risk_report bundle -------------------------------------------------------

def test_risk_report_keys_and_consistency() -> None:
    report = risk_report(SYMMETRIC, confidence=0.90)
    assert set(report) == {
        "var_historical",
        "es_historical",
        "var_parametric",
        "es_parametric",
    }
    # Bundle values match the standalone calls, and ES >= VaR holds per method.
    assert report["var_historical"] == pytest.approx(value_at_risk(SYMMETRIC, 0.90))
    assert report["es_historical"] == pytest.approx(expected_shortfall(SYMMETRIC, 0.90))
    assert report["var_parametric"] == pytest.approx(
        value_at_risk(SYMMETRIC, 0.90, "parametric")
    )
    assert report["es_parametric"] == pytest.approx(
        expected_shortfall(SYMMETRIC, 0.90, "parametric")
    )
    assert report["es_historical"] >= report["var_historical"]
    assert report["es_parametric"] >= report["var_parametric"]
