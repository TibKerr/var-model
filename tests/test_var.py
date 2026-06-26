"""Tests for historical Value-at-Risk (the five pillars).

1. Reference value   — a hand-computable quantile with no interpolation.
2. Cross-check       — large normal sample vs. the theoretical normal quantile
                       (scipy), independent of our own parametric code.
3. Invariants        — VaR monotone in confidence; horizon/value scaling.
4. Edge cases        — degenerate (constant) returns.
5. Validation        — bad inputs raise ValueError naming the argument.
"""

import numpy as np
import pytest
from scipy.stats import norm

from var_model import value_at_risk

# 11 evenly spaced returns from -0.05 to +0.05. With linear interpolation the
# quantile position is alpha*(n-1) = alpha*10, an integer for alpha in {0.1, 0.2},
# so these quantiles land exactly on an order statistic (no interpolation).
SYMMETRIC = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


# --- Pillar 1: reference value ------------------------------------------------

def test_reference_value_exact_quantile() -> None:
    # confidence 0.90 -> alpha 0.10 -> position 1.0 -> 2nd-worst return = -0.04.
    assert value_at_risk(SYMMETRIC, confidence=0.90) == pytest.approx(0.04)


def test_reference_value_second_level() -> None:
    # confidence 0.80 -> alpha 0.20 -> position 2.0 -> 3rd-worst return = -0.03.
    assert value_at_risk(SYMMETRIC, confidence=0.80) == pytest.approx(0.03)


# --- Pillar 2: independent cross-check ----------------------------------------

def test_matches_theoretical_normal_quantile() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.0, scale=0.02, size=200_000)
    historical = value_at_risk(sample, confidence=0.99)
    theoretical = norm.ppf(0.99) * 0.02  # positive loss at the 99% level
    assert historical == pytest.approx(theoretical, rel=0.05)


# --- Pillar 3: invariants -----------------------------------------------------

def test_var_monotone_in_confidence() -> None:
    assert value_at_risk(SYMMETRIC, confidence=0.99) >= value_at_risk(
        SYMMETRIC, confidence=0.95
    )


def test_horizon_scales_as_sqrt_time() -> None:
    one_day = value_at_risk(SYMMETRIC, confidence=0.95, horizon=1)
    four_day = value_at_risk(SYMMETRIC, confidence=0.95, horizon=4)
    assert four_day == pytest.approx(2.0 * one_day)


def test_value_scales_linearly() -> None:
    frac = value_at_risk(SYMMETRIC, confidence=0.90)
    dollars = value_at_risk(SYMMETRIC, confidence=0.90, value=1_000_000)
    assert dollars == pytest.approx(frac * 1_000_000)


# --- Pillar 4: edge cases -----------------------------------------------------

def test_zero_returns_give_zero_var() -> None:
    assert value_at_risk([0.0] * 50, confidence=0.95) == pytest.approx(0.0)


def test_constant_loss_returns_that_constant() -> None:
    # Every observation is -1%; the quantile is -0.01 -> VaR = +0.01.
    assert value_at_risk([-0.01] * 50, confidence=0.95) == pytest.approx(0.01)


# --- Pillar 5: validation -----------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": 1.5},
        {"horizon": 0},
        {"value": 0.0},
        {"method": "bogus"},
    ],
)
def test_validation_rejects_bad_scalars(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        value_at_risk(SYMMETRIC, **kwargs)  # type: ignore[arg-type]


def test_validation_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="returns"):
        value_at_risk([], confidence=0.95)


def test_validation_rejects_nonfinite_returns() -> None:
    with pytest.raises(ValueError, match="finite"):
        value_at_risk([0.01, np.nan, -0.02], confidence=0.95)


# =============================================================================
# Parametric (variance-covariance) VaR
# =============================================================================

# mean 0, unbiased (ddof=1) std = sqrt(2.5) * 0.01.
NORMAL5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 0.01


# --- Pillar 1: reference value (closed-form normal formula) -------------------

def test_parametric_reference_normal_formula() -> None:
    # z*sigma - mu with mu = 0; sigma written out as the exact ddof=1 std.
    expected = norm.ppf(0.99) * np.sqrt(2.5) * 0.01
    assert value_at_risk(NORMAL5, confidence=0.99, method="parametric") == pytest.approx(
        expected, rel=1e-12
    )


def test_parametric_incorporates_mean() -> None:
    # Same shape shifted up by 0.5%: sigma unchanged, mu = 0.005 reduces the loss.
    shifted = NORMAL5 + 0.005
    expected = norm.ppf(0.99) * np.sqrt(2.5) * 0.01 - 0.005
    assert value_at_risk(shifted, confidence=0.99, method="parametric") == pytest.approx(
        expected, rel=1e-12
    )


# --- Pillar 2: independent cross-check (THE headline: parametric vs historical) ---

def test_parametric_agrees_with_historical_on_normal() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=0.0005, scale=0.02, size=200_000)
    for c in (0.95, 0.99):
        parametric = value_at_risk(sample, confidence=c, method="parametric")
        historical = value_at_risk(sample, confidence=c, method="historical")
        assert parametric == pytest.approx(historical, rel=0.05)


# --- Pillar 3: invariants -----------------------------------------------------

def test_parametric_monotone_in_confidence() -> None:
    assert value_at_risk(NORMAL5, confidence=0.99, method="parametric") >= value_at_risk(
        NORMAL5, confidence=0.95, method="parametric"
    )


def test_parametric_horizon_and_value_scaling() -> None:
    base = value_at_risk(NORMAL5, confidence=0.95, method="parametric")
    scaled = value_at_risk(
        NORMAL5, confidence=0.95, method="parametric", horizon=4, value=1_000_000
    )
    assert scaled == pytest.approx(base * 2.0 * 1_000_000)


# --- Pillar 4: edge cases -----------------------------------------------------

def test_parametric_zero_variance_zero_returns() -> None:
    # sigma = 0, mu = 0 -> VaR = 0.
    assert value_at_risk([0.0] * 10, confidence=0.95, method="parametric") == pytest.approx(
        0.0
    )


def test_parametric_constant_loss_is_that_loss() -> None:
    # sigma = 0, mu = -0.01 -> VaR = -mu = 0.01 (a certain 1% loss).
    assert value_at_risk(
        [-0.01] * 10, confidence=0.95, method="parametric"
    ) == pytest.approx(0.01)


# --- Pillar 5: validation -----------------------------------------------------

def test_parametric_requires_at_least_two_returns() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        value_at_risk([0.01], confidence=0.95, method="parametric")
