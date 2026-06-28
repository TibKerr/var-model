"""Command-line interface for var_model.

Thin by design: it parses arguments, calls the pipeline, and formats output.
All business logic lives in the core and data layers.

Commands:
- ``run``     fetch (or reuse cached) prices, compute VaR/ES three ways, store
              the result, and print the comparison.
- ``history`` list previously stored runs.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from var_model import __version__
from var_model.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON,
    DEFAULT_N_SIMS,
    DEFAULT_TICKERS,
    DEFAULT_VALUE,
    DEFAULT_WINDOW,
)
from var_model.data import (
    Run,
    init_db,
    load_runs,
    make_engine,
    run_portfolio_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="var-model",
        description="Value-at-Risk and Expected Shortfall for an equity portfolio.",
    )
    parser.add_argument("--version", action="version", version=f"var-model {__version__}")

    # Shared options available on each subcommand (e.g. `run --db ...`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=None, help="database URL (default: env or local SQLite)")

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        "run",
        parents=[common],
        help="fetch, compute, store, and print a risk comparison",
    )
    run.add_argument("-t", "--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    run.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE)
    run.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    run.add_argument("--value", type=float, default=DEFAULT_VALUE)
    run.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    run.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS, dest="n_sims")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--label", default=None)
    run.add_argument(
        "--no-fetch",
        action="store_false",
        dest="fetch",
        help="use only cached prices; do not call Alpha Vantage",
    )

    sub.add_parser("history", parents=[common], help="list previously stored runs")
    return parser


def _print_run(run: Run) -> None:
    print(f"Run #{run.id}  {run.label or ''}".rstrip())
    print(
        f"  confidence={run.confidence:.2%}  horizon={run.horizon}d  "
        f"value={run.value:,.0f}  n_obs={run.n_observations}"
    )
    print(f"  {'method':<12} {'VaR':>14} {'ES':>14}")
    for result in sorted(run.results, key=lambda r: r.method):
        print(f"  {result.method:<12} {result.var:>14,.2f} {result.es:>14,.2f}")
    print(
        f"  spread: VaR {run.var_spread:,.2f} ({run.var_spread_relative:.1%})  "
        f"ES {run.es_spread:,.2f} ({run.es_spread_relative:.1%})"
    )
    verdict = (
        "normality rejected -> methods expected to diverge"
        if run.jarque_bera_pvalue < 0.05
        else "normality not rejected -> methods expected to agree"
    )
    print(
        f"  diagnostics: skew={run.skewness:+.3f}  excess_kurtosis="
        f"{run.excess_kurtosis:+.3f}  JB p={run.jarque_bera_pvalue:.3f}  ({verdict})"
    )


def _cmd_run(args: argparse.Namespace) -> None:
    engine = make_engine(args.db)
    init_db(engine)
    with Session(engine) as session:
        run = run_portfolio_analysis(
            session,
            args.tickers,
            fetch=args.fetch,
            confidence=args.confidence,
            horizon=args.horizon,
            value=args.value,
            window=args.window,
            n_sims=args.n_sims,
            seed=args.seed,
            label=args.label,
        )
        _print_run(run)


def _cmd_history(args: argparse.Namespace) -> None:
    engine = make_engine(args.db)
    init_db(engine)
    with Session(engine) as session:
        runs = load_runs(session)
        if not runs:
            print("No runs stored yet.")
            return
        for run in runs:
            _print_run(run)
            print()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to a subcommand."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _cmd_run(args)
    elif args.command == "history":
        _cmd_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
