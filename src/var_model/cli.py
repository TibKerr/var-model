"""Command-line interface for var_model.

This is a thin entry point: it parses arguments and delegates. Domain commands
(data fetch, VaR computation, divergence report) are wired in as later phases
land their logic. For now it exposes ``--version`` so the scaffold is runnable
and CI has something to exercise.
"""

import argparse

from var_model import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="var-model",
        description="Value-at-Risk and Expected Shortfall for an equity portfolio.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"var-model {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch.

    With no subcommands wired yet, invoking the CLI without arguments prints
    help so the command is self-describing.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # No subcommands yet; show help so the command is never a silent no-op.
    _ = args
    parser.print_help()


if __name__ == "__main__":
    main()
