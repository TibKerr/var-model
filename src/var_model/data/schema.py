"""SQLAlchemy models for persisted risk results.

Two tables, normalized so runs are comparable across time:

- ``runs`` — one row per analysis: the parameters it was run with, the
  distribution diagnostics, and the between-method spreads.
- ``method_results`` — three rows per run (historical, parametric, monte_carlo),
  each holding that method's VaR and ES.

These are pure data definitions; engine/session handling lives in ``database``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC)
    )

    # Parameters the analysis was run with.
    confidence: Mapped[float]
    horizon: Mapped[int]
    value: Mapped[float]
    n_sims: Mapped[int]
    seed: Mapped[int | None] = mapped_column(default=None)
    n_observations: Mapped[int] = mapped_column(default=0)
    label: Mapped[str | None] = mapped_column(String(255), default=None)

    # Distribution diagnostics (why the methods diverge, or don't).
    mean: Mapped[float] = mapped_column(default=0.0)
    std: Mapped[float] = mapped_column(default=0.0)
    skewness: Mapped[float] = mapped_column(default=0.0)
    excess_kurtosis: Mapped[float] = mapped_column(default=0.0)
    jarque_bera: Mapped[float] = mapped_column(default=0.0)
    jarque_bera_pvalue: Mapped[float] = mapped_column(default=1.0)

    # Between-method spread (how far apart the methods landed).
    var_spread: Mapped[float] = mapped_column(default=0.0)
    var_spread_relative: Mapped[float] = mapped_column(default=0.0)
    es_spread: Mapped[float] = mapped_column(default=0.0)
    es_spread_relative: Mapped[float] = mapped_column(default=0.0)

    results: Mapped[list[MethodResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"Run(id={self.id!r}, confidence={self.confidence!r}, "
            f"n_observations={self.n_observations!r}, label={self.label!r})"
        )


class MethodResult(Base):
    __tablename__ = "method_results"
    __table_args__ = (UniqueConstraint("run_id", "method", name="uq_run_method"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    method: Mapped[str] = mapped_column(String(20))
    var: Mapped[float]
    es: Mapped[float]

    run: Mapped[Run] = relationship(back_populates="results")

    def __repr__(self) -> str:
        return (
            f"MethodResult(method={self.method!r}, var={self.var!r}, es={self.es!r})"
        )
