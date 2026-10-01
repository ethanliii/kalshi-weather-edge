"""Render reports/results.json as Markdown tables (reports/RESULTS.md).

Generating the tables from the JSON keeps every number in the write-up traceable
to the evaluation run that produced it.
"""

from __future__ import annotations

NAMES = {"p_emos": "EMOS model", "p_raw": "Raw ensemble", "p_market": "Market (mid)"}


def _ci(e: dict, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(e['mean'])} [{fmt.format(e['lo'])}, {fmt.format(e['hi'])}]"


def _money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def render(results: dict) -> str:
    out = [f"# Results\n\nGenerated {results['generated_utc']} by `weather-edge evaluate`. "
           "Intervals are 95% bootstrap CIs resampling dates.\n"]

    out.append("## Settlement value vs NWS CLI\n")
    out.append("| Settlement source | Events | Exact match | Within 1°F |\n|---|---|---|---|")
    for src, r in results["settlement_check"].items():
        out.append(f"| {src} | {r['n_events']:,} | {r['exact_match']:.2%} | "
                   f"{r['within_1F']:.2%} |")

    for lead, r in results["leads"].items():
        fs, vm = r["forecast_skill"], r["vs_market"]
        out.append(f"\n## Lead {lead}\n")
        out.append(f"### Forecast skill vs observed CLI high ({fs['n_station_days']:,} "
                   f"station-days, {fs['n_stations']} stations, {fs['period'][0]} to "
                   f"{fs['period'][1]})\n")
        out.append("| | EMOS | Raw multi-model Gaussian |\n|---|---|---|")
        out.append(f"| CRPS (°F) | {_ci(fs['crps_emos'])} | {_ci(fs['crps_raw_gaussian'])} |")
        out.append(f"| MAE of mean (°F) | {_ci(fs['mae_emos_mean'])} | "
                   f"{_ci(fs['mae_raw_mean'])} |")
        out.append(f"| 80% interval coverage | {fs['coverage_80_emos']:.1%} | "
                   f"{fs['coverage_80_raw']:.1%} |")

        out.append(f"\n### Probabilities vs market ({vm['n_events']:,} events, "
                   f"{len(vm['series'])} series, {vm['period'][0]} to {vm['period'][1]})\n")
        out.append("| Forecaster | Brier | Log loss | RPS |\n|---|---|---|---|")
        for f in ("p_emos", "p_raw", "p_market"):
            out.append(f"| {NAMES[f]} | {_ci(vm['brier'][f])} | {_ci(vm['log_loss'][f])} | "
                       f"{_ci(vm['rps'][f])} |")
        out.append(f"| **EMOS − market** | {_ci(vm['brier']['emos_minus_market'])} | "
                   f"{_ci(vm['log_loss']['emos_minus_market'])} | "
                   f"{_ci(vm['rps']['emos_minus_market'])} |")
        dropped = vm.get("dropped_events", {})
        out.append(f"\nEvents excluded: {dropped.get('bad_quotes', 0):,} with missing, stale "
                   "(>3 h) or one-sided quotes; events before the first out-of-sample month "
                   "have no forecast.\n")

        out.append("Log loss, EMOS − market, by series (negative = model better):\n")
        out.append("| Series | Events | Δ log loss |\n|---|---|---|")
        for s, d in sorted(vm["log_loss_emos_minus_market_by_series"].items()):
            out.append(f"| {s} | {d['n']:,} | {_ci(d)} |")

        out.append("\n### Backtest (10 contracts per signal, taker, fees and spread included)\n")
        out.append("| Margin | Trades | Days | Fees | P&L | ROI on capital | P&L per trade |"
                   "\n|---|---|---|---|---|---|---|")
        for key, b in r["backtest"].items():
            if b is None:
                out.append(f"| {key[7:]} | 0 | | | | | |")
                continue
            out.append(f"| {key[7:]} | {b['n_trades']:,} | {b['n_days']:,} | "
                       f"{_money(b['fees_paid'])} | {_money(b['total_pnl'])} | "
                       f"{_ci(b['roi'], '{:+.1%}')} | {_ci(b['pnl_per_trade'], '{:+.3f}')} |")

        ab = r.get("fresh_run_ablation")
        if ab:
            cities = ", ".join(ab["series"])
            out.append(f"\n### Ablation: add the decision-day 00Z ECMWF run ({cities}; "
                       f"{ab['n_events']:,} events)\n")
            out.append("| | Base EMOS | + fresh run | Market | Fresh − base | Fresh − market |"
                       "\n|---|---|---|---|---|---|")
            out.append(f"| CRPS (°F) | {_ci(ab['crps_base'])} | {_ci(ab['crps_fresh'])} | | | |")
            for m in ("brier", "log_loss"):
                x = ab[m]
                out.append(f"| {m.replace('_', ' ').title()} | {_ci(x['base'])} | "
                           f"{_ci(x['fresh'])} | {_ci(x['market'])} | "
                           f"{_ci(x['fresh_minus_base'])} | {_ci(x['fresh_minus_market'])} |")
            b = ab.get("backtest_margin_0.03")
            if b:
                out.append(f"\nBacktest with the fresh model (margin 0.03): {b['n_trades']:,} "
                           f"trades, P&L {_money(b['total_pnl'])}, ROI "
                           f"{_ci(b['roi'], '{:+.1%}')}.")
    return "\n".join(out) + "\n"
