"""Tests for the SQL persistence layer (in-memory SQLite round-trips)."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from var_model import divergence_report
from var_model.data import (
    MethodResult,
    Run,
    compute_and_save,
    init_db,
    load_runs,
    save_divergence_report,
)

METHODS = {"historical", "parametric", "monte_carlo"}


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def _returns(seed: int = 0, size: int = 2_000) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0003, 0.02, size)


def test_save_persists_run_and_three_results() -> None:
    report = divergence_report(_returns(), confidence=0.99, value=1_000_000, seed=0)
    with _session() as session:
        run = save_divergence_report(
            session, report, confidence=0.99, value=1_000_000, seed=0, label="unit-test"
        )
        run_id = run.id
        # Force a reload from the database rather than the identity map.
        session.expire_all()
        loaded = session.get(Run, run_id)
        assert loaded is not None
        assert loaded.confidence == 0.99
        assert loaded.value == 1_000_000
        assert loaded.label == "unit-test"
        assert loaded.created_at is not None
        assert {r.method for r in loaded.results} == METHODS


def test_persisted_values_match_the_report() -> None:
    report = divergence_report(_returns(), confidence=0.95, seed=0)
    with _session() as session:
        run = save_divergence_report(session, report, confidence=0.95, seed=0)
        session.expire_all()
        loaded = session.get(Run, run.id)
        assert loaded is not None
        # Run-level diagnostics and spreads round-trip exactly.
        assert loaded.excess_kurtosis == pytest.approx(report["excess_kurtosis"])
        assert loaded.jarque_bera_pvalue == pytest.approx(report["jarque_bera_pvalue"])
        assert loaded.var_spread == pytest.approx(report["var_spread"])
        assert loaded.n_observations == int(report["n_observations"])
        # Per-method VaR/ES round-trip exactly.
        by_method = {r.method: r for r in loaded.results}
        for m in METHODS:
            assert by_method[m].var == pytest.approx(report[f"var_{m}"])
            assert by_method[m].es == pytest.approx(report[f"es_{m}"])


def test_compute_and_save_matches_direct_report() -> None:
    returns = _returns()
    with _session() as session:
        run = compute_and_save(session, returns, confidence=0.99, seed=0, label="e2e")
        report = divergence_report(returns, confidence=0.99, seed=0)
        by_method = {r.method: r for r in run.results}
        assert by_method["historical"].var == pytest.approx(report["var_historical"])
        assert run.excess_kurtosis == pytest.approx(report["excess_kurtosis"])


def test_load_runs_returns_all_in_order() -> None:
    with _session() as session:
        compute_and_save(session, _returns(1), confidence=0.95, seed=0, label="a")
        compute_and_save(session, _returns(2), confidence=0.99, seed=0, label="b")
        runs = load_runs(session)
        assert [r.label for r in runs] == ["a", "b"]
        assert all(len(r.results) == 3 for r in runs)


def test_duplicate_method_violates_unique_constraint() -> None:
    with _session() as session:
        run = Run(confidence=0.95, horizon=1, value=1.0, n_sims=100, n_observations=10)
        run.results = [
            MethodResult(method="historical", var=0.01, es=0.02),
            MethodResult(method="historical", var=0.03, es=0.04),
        ]
        session.add(run)
        with pytest.raises(IntegrityError):
            session.commit()
