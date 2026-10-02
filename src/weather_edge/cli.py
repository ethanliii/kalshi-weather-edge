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
        if "fresh" in what:
            collect.collect_fresh_runs(st)
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


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit EMOS on all history and write models/emos_params.json (used by live + CI)."""
    from weather_edge import live

    params = live.fit_all(args.series or sorted(STATIONS))
    live.save_params(params)
    print(f"wrote {live.PARAMS_PATH} ({len(params['models'])} models)")


def cmd_daily(args: argparse.Namespace) -> None:
    """Forward test: refresh weather, snapshot fair values vs live quotes, score settled."""
    from weather_edge import daily

    daily.run()


def _run_trader(args: argparse.Namespace, shadow: bool) -> None:
    import time

    from weather_edge.config import LOG_DIR
    from weather_edge.kalshi.auth import KeyFileError
    from weather_edge.trading.demo_client import (
        DemoClient,
        MissingCredentials,
        ProductionReadOnlyClient,
    )
    from weather_edge.trading.paper import PaperTrader
    from weather_edge.trading.risk import KillSwitch, KillSwitchConfig, Ledger
    from weather_edge.trading.sizing import SizingConfig

    state = LOG_DIR / ("shadow" if shadow else "paper")
    ledger = Ledger(state / "ledger.jsonl")
    kill = KillSwitch(state, KillSwitchConfig(max_daily_loss=args.max_daily_loss), ledger)
    sizing = SizingConfig(kelly_fraction=args.kelly, min_edge=args.margin,
                          max_market_exposure=args.max_market,
                          max_event_exposure=args.max_event, max_total_exposure=args.max_total)
    try:
        client = ProductionReadOnlyClient() if shadow else DemoClient()
    except (MissingCredentials, KeyFileError) as exc:
        which = "production" if shadow else "demo"
        raise SystemExit(f"needs valid {which} API keys: {exc}") from None
    trader = PaperTrader(client, ledger, kill, sizing, series=args.series, shadow=shadow,
                         bankroll_override=getattr(args, "bankroll", None))
    verb = "would-be orders logged (nothing sent)" if shadow else "orders sent"
    while True:
        try:
            orders = trader.run_once()
            print(f"{len(orders)} {verb}; ledger: {ledger.path}")
        except Exception as exc:  # already counted by the kill switch and logged to the ledger
            logging.getLogger(__name__).error("run failed: %s", exc)
            if not args.loop:
                raise SystemExit(1) from exc
        if not args.loop or kill.tripped:
            break
        time.sleep(args.interval)


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trade on the Kalshi DEMO exchange (never production)."""
    _run_trader(args, shadow=False)


def cmd_shadow(args: argparse.Namespace) -> None:
    """Shadow mode on PRODUCTION: real prices and balance, orders logged but never sent."""
    _run_trader(args, shadow=True)


def cmd_shadow_report(args: argparse.Namespace) -> None:
    from weather_edge.config import LOG_DIR
    from weather_edge.trading import shadow_report
    from weather_edge.trading.risk import Ledger

    df = shadow_report.score(Ledger(LOG_DIR / "shadow" / "ledger.jsonl"))
    print(shadow_report.summary(df))
    if args.csv and not df.empty:
        df.to_csv(args.csv, index=False)
        print(f"details: {args.csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="weather-edge")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("markets", help="show open temperature events and quotes")
    m.add_argument("--series", nargs="*", help="series tickers (default: all)")
    m.set_defaults(func=cmd_markets)

    c = sub.add_parser("collect", help="fetch and cache raw data")
    c.add_argument("what", nargs="+", choices=["weather", "markets", "quotes", "ensemble", "fresh"])
    c.add_argument("--series", nargs="*", help="series tickers (default: all)")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("evaluate", help="fit models out of sample and write reports/")
    e.add_argument("--leads", nargs="*", type=int, default=[1, 2])
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser("fit", help="fit EMOS on all history -> models/emos_params.json")
    f.add_argument("--series", nargs="*", help="series tickers (default: all)")
    f.set_defaults(func=cmd_fit)

    d = sub.add_parser("daily", help="forward-test snapshot (read-only; used by CI)")
    d.set_defaults(func=cmd_daily)

    def trader_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--series", nargs="*", help="series tickers (default: all)")
        sp.add_argument("--margin", type=float, default=0.03, help="edge required after fees ($)")
        sp.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly")
        sp.add_argument("--max-market", type=float, default=25.0, help="$ at risk per market")
        sp.add_argument("--max-event", type=float, default=50.0, help="$ at risk per event")
        sp.add_argument("--max-total", type=float, default=250.0, help="$ at risk in total")
        sp.add_argument("--max-daily-loss", type=float, default=100.0,
                        help="kill-switch loss limit")
        sp.add_argument("--loop", action="store_true", help="keep running every --interval s")
        sp.add_argument("--interval", type=int, default=900)

    pt = sub.add_parser("paper", help="paper trade on the Kalshi DEMO exchange")
    trader_args(pt)
    pt.set_defaults(func=cmd_paper)

    sh = sub.add_parser("shadow", help="PRODUCTION shadow mode: real prices, orders logged only")
    trader_args(sh)
    sh.add_argument("--bankroll", type=float, default=None,
                    help="size as if the account held this many $ (default: real balance)")
    sh.set_defaults(func=cmd_shadow)

    sr = sub.add_parser("shadow-report", help="score shadow orders against settlements")
    sr.add_argument("--csv", help="also write per-order details to this CSV")
    sr.set_defaults(func=cmd_shadow_report)
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
