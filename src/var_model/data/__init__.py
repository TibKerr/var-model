"""I/O layer for var_model: SQL persistence of risk/divergence results.

This package owns everything that touches the database. The math core
(``var.py``, ``risk.py``, ``divergence.py``) stays pure and never imports from
here; persistence consumes the plain result dicts the core produces.
"""

from var_model.data.schema import Base, MethodResult, Run

__all__ = [
    "Base",
    "MethodResult",
    "Run",
]
