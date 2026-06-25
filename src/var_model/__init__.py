"""var_model: Value-at-Risk and Expected Shortfall for an equity portfolio.

The public API surface is intentionally small and grows as phases land:
- Phase 3 adds the VaR core (``value_at_risk``).
- Phase 4 adds Expected Shortfall and the ``risk_report`` bundle helper.
- Phase 5 adds the divergence analysis.
"""

__version__ = "0.1.0"

__all__: list[str] = [
    "__version__",
]
