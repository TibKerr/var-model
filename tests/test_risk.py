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
    report = risk_report(SYMMETRIC, confidence=0.90, seed=0)
    assert set(report) == {
        "var_historical",
        "es_historical",
        "var_parametric",
        "es_parametric",
        "var_monte_carlo",
        "es_monte_carlo",
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
    assert report["var_monte_carlo"] == pytest.approx(
        value_at_risk(SYMMETRIC, 0.90, "monte_carlo", seed=0)
    )
    assert report["es_monte_carlo"] == pytest.approx(
        expected_shortfall(SYMMETRIC, 0.90, "monte_carlo", seed=0)
    )
    assert report["es_historical"] >= report["var_historical"]
    assert report["es_parametric"] >= report["var_parametric"]
    assert report["es_monte_carlo"] >= report["var_monte_carlo"]


# =============================================================================
# Parametric (variance-covariance) Expected Shortfall
# =============================================================================

# mean 0, unbiased (ddof=1) std = sqrt(2.5) * 0.01.
NORMAL5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 0.01


def test_parametric_es_reference_normal_formula() -> None:
    sigma = np.sqrt(2.5) * 0.01
    c = 0.99
    expected = sigma * norm.pdf(norm.ppf(c)) / (1.0 - c)  # mu = 0
    assert expected_shortfall(NORMAL5, confidence=c, method="parametric") == pytest.approx(
        expected, rel=1e-12
    )


def test_parametric_es_incorporates_mean() -> None:
    shifted = NORMAL5 + 0.005
    sigma = np.sqrt(2.5) * 0.01
    c = 0.99
    expected = sigma * norm.pdf(norm.ppf(c)) / (1.0 - c) - 0.005
    assert expected_shortfall(shifted, confidence=c, method="parametric") == pytest.approx(
        expected, rel=1e-12
    )


def test_parametric_es_agrees_with_historical_on_normal() -> None:
    rng = np.random.default_rng(11)
    sample = rng.normal(loc=0.0, scale=0.02, size=300_000)
    for c in (0.95, 0.99):
        parametric = expected_shortfall(sample, confidence=c, method="parametric")
        historical = expected_shortfall(sample, confidence=c, method="historical")
        assert parametric == pytest.approx(historical, rel=0.05)


def test_parametric_es_at_least_var() -> None:
    rng = np.random.default_rng(12)
    sample = rng.normal(0.0, 0.02, 50_000)
    for c in (0.90, 0.95, 0.99):
        es = expected_shortfall(sample, confidence=c, method="parametric")
        var = value_at_risk(sample, confidence=c, method="parametric")
        assert es >= var


def test_parametric_es_requires_at_least_two_returns() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        expected_shortfall([0.01], confidence=0.95, method="parametric")


# =============================================================================
# Monte Carlo Expected Shortfall
# =============================================================================

def test_mc_es_reference_normal_tail() -> None:
    sigma = np.sqrt(2.5) * 0.01
    c = 0.99
    expected = sigma * norm.pdf(norm.ppf(c)) / (1.0 - c)  # mu = 0
    mc = expected_shortfall(NORMAL5, confidence=c, method="monte_carlo", n_sims=200_000, seed=0)
    assert mc == pytest.approx(expected, rel=0.05)


def test_mc_es_converges_to_parametric() -> None:
    rng = np.random.default_rng(13)
    sample = rng.normal(0.0, 0.02, 5_000)
    for c in (0.95, 0.99):
        mc = expected_shortfall(sample, confidence=c, method="monte_carlo", n_sims=200_000, seed=0)
        parametric = expected_shortfall(sample, confidence=c, method="parametric")
        assert mc == pytest.approx(parametric, rel=0.04)


def test_mc_es_at_least_var_shared_seed() -> None:
    rng = np.random.default_rng(14)
    sample = rng.normal(0.0, 0.02, 5_000)
    for c in (0.90, 0.95, 0.99):
        es = expected_shortfall(sample, confidence=c, method="monte_carlo", seed=5)
        var = value_at_risk(sample, confidence=c, method="monte_carlo", seed=5)
        assert es >= var


def test_mc_es_reproducible_with_seed() -> None:
    a = expected_shortfall(NORMAL5, confidence=0.95, method="monte_carlo", seed=7)
    b = expected_shortfall(NORMAL5, confidence=0.95, method="monte_carlo", seed=7)
    assert a == b


def test_mc_es_requires_positive_n_sims() -> None:
    with pytest.raises(ValueError, match="n_sims"):
        expected_shortfall(NORMAL5, confidence=0.95, method="monte_carlo", n_sims=0, seed=0)
