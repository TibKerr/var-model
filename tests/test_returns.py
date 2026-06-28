"""Tests for price -> log return transformation and portfolio aggregation."""

import numpy as np
import pandas as pd
import pytest

from var_model.data import log_returns, portfolio_returns


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=3)
    # A compounds +10% each step; B is flat.
    return pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 50.0]}, index=idx)


def test_log_returns_values() -> None:
    lr = log_returns(_prices())
    assert lr.shape == (2, 2)
    assert lr["A"].iloc[0] == pytest.approx(np.log(1.1))
    assert lr["A"].iloc[1] == pytest.approx(np.log(1.1))
    assert lr["B"].iloc[0] == pytest.approx(0.0)


def test_log_returns_drops_first_row_keeps_dates() -> None:
    prices = _prices()
    lr = log_returns(prices)
    assert list(lr.index) == list(prices.index[1:])


def test_log_returns_requires_two_rows() -> None:
    one = pd.DataFrame({"A": [100.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="at least 2"):
        log_returns(one)


def test_log_returns_rejects_nonpositive_prices() -> None:
    bad = pd.DataFrame({"A": [100.0, 0.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="positive"):
        log_returns(bad)


def test_portfolio_equal_weight_is_row_mean() -> None:
    df = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
    assert portfolio_returns(df) == pytest.approx([0.2, 0.3])


def test_portfolio_custom_weights() -> None:
    df = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
    out = portfolio_returns(df, weights=[0.25, 0.75])
    assert out == pytest.approx([0.1 * 0.25 + 0.3 * 0.75, 0.2 * 0.25 + 0.4 * 0.75])


def test_portfolio_weight_length_mismatch() -> None:
    df = pd.DataFrame({"A": [0.1], "B": [0.2]})
    with pytest.raises(ValueError, match="assets"):
        portfolio_returns(df, weights=[1.0])


def test_portfolio_weights_must_sum_to_one() -> None:
    df = pd.DataFrame({"A": [0.1], "B": [0.2]})
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_returns(df, weights=[0.5, 0.4])


def test_portfolio_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        portfolio_returns(pd.DataFrame())
