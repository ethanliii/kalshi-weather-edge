"""Command-line entry point: `weather-edge <command>` (or `python -m weather_edge.cli`)."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from weather_edge import collect
from weather_edge.kalshi.client import KalshiClient
from weather_edge.kalshi.markets import sorted_brackets, validate_partition
from weather_edge.stations import STATIONS


def cmd_markets(args: argparse.Namespace) -> None:
    """Print open events with bracket quotes and check each bracket set is a partition."""
    client = KalshiClient()
    series = args.series or sorted(STATIONS)
    for s in series:
        events: dict[str, list[dict]] = defaultdict(list)
        for m in client.open_markets(s):
            events[m["event_ticker"]].append(m)
        for ev, ms in sorted(events.items()):
            brackets = sorted_brackets(ms)
            try:
                validate_partition(brackets)
                ok = "ok"
            except ValueError as exc:
                ok = f"INVALID: {exc}"
            by_ticker = {m["ticker"]: m for m in ms}
            quotes = " ".join(
                f"{b.label}:{float(by_ticker[b.ticker]['yes_bid_dollars']):.2f}"
                f"/{float(by_ticker[b.ticker]['yes_ask_dollars']):.2f}"
                for b in brackets
            )
            print(f"{ev:<22} {ok:<4} {quotes}")


def cmd_collect(args: argparse.Namespace) -> None:
    """Fetch and cache raw data. Incremental: safe to re-run."""
    stations = collect.stations_for(args.series)
    what = set(args.what)
    client = KalshiClient()
    for st in stations:
        if "weather" in what:
            collect.collect_cli(st)
            collect.collect_forecasts(st)
        if "markets" in what:
            collect.collect_markets(client, st)
        if "quotes" in what:
            collect.collect_quotes(client, st)
    if "ensemble" in what:
        collect.collect_ensemble_snapshot(stations)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Walk-forward EMOS, model-vs-market scores, backtest; writes reports/."""
    from weather_edge.evaluation import report

    results = report.run(tuple(args.leads))
    for lead, r in results["leads"].items():
        vm = r["vs_market"]
        ll = vm["log_loss"]
        print(f"lead {lead}: {vm['n_events']} events | log loss EMOS {ll['p_emos']['mean']:.3f} "
              f"market {ll['p_market']['mean']:.3f} | diff {ll['emos_minus_market']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="weather-edge")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("markets", help="show open temperature events and quotes")
    m.add_argument("--series", nargs="*", help="series tickers (default: all)")
    m.set_defaults(func=cmd_markets)

    c = sub.add_parser("collect", help="fetch and cache raw data")
    c.add_argument("what", nargs="+", choices=["weather", "markets", "quotes", "ensemble"])
    c.add_argument("--series", nargs="*", help="series tickers (default: all)")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("evaluate", help="fit models out of sample and write reports/")
    e.add_argument("--leads", nargs="*", type=int, default=[1, 2])
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
