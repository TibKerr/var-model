"""var_model: Value-at-Risk and Expected Shortfall for an equity portfolio.

The public API surface:
- VaR (``value_at_risk``) and Expected Shortfall (``expected_shortfall``,
  ``risk_report``) across all three methods.
- The comparative divergence analysis (``divergence_report``,
  ``distribution_diagnostics``).
"""

from var_model.divergence import distribution_diagnostics, divergence_report
from var_model.risk import expected_shortfall, risk_report
from var_model.var import Method, validate_inputs, value_at_risk

__version__ = "0.1.0"

__all__: list[str] = [
    "__version__",
    "Method",
    "validate_inputs",
    "value_at_risk",
    "expected_shortfall",
    "risk_report",
    "distribution_diagnostics",
    "divergence_report",
]
