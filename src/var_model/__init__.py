"""var_model: Value-at-Risk and Expected Shortfall for an equity portfolio.

The public API surface grows as each method milestone lands:
- Historical VaR (``value_at_risk``) — done.
- Expected Shortfall and the ``risk_report`` bundle helper — next.
- Parametric and Monte Carlo methods, then the divergence analysis.
"""

from var_model.var import Method, validate_inputs, value_at_risk

__version__ = "0.1.0"

__all__: list[str] = [
    "__version__",
    "Method",
    "validate_inputs",
    "value_at_risk",
]
