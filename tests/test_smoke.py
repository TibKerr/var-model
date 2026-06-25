"""Smoke tests for the Phase 1 scaffold.

These prove the package imports cleanly under the src layout and the CLI is
wired through pyproject correctly. Domain tests (the five pillars) arrive with
their modules in later phases.
"""

import var_model
from var_model.cli import build_parser, main


def test_package_imports_and_exposes_version() -> None:
    assert isinstance(var_model.__version__, str)
    assert var_model.__version__ == "0.1.0"


def test_cli_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "var-model"


def test_cli_runs_without_arguments() -> None:
    # Should print help and return cleanly (no SystemExit, no crash).
    main([])
