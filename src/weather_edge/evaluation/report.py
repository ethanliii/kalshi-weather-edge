"""End-to-end evaluation: forecast skill, model vs market, backtest, figures.

Writes reports/results.json (compact, committed) and reports/figures/*.png.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from weather_edge import dataset
from weather_edge.config import FIGURES_DIR, REPORTS_DIR
from weather_edge.evaluation import plots
from weather_edge.evaluation.backtest import BacktestConfig, simulate, summarise
from weather_edge.evaluation.metrics import Estimate, cluster_bootstrap, event_scores
from weather_edge.model.emos import crps_normal

log = logging.getLogger(__name__)
LEADS = (1, 2)
FORECASTERS = ("p_emos", "p_raw", "p_market")
MARGINS = (0.0, 0.02, 0.03, 0.05, 0.10)


def _est(e: Estimate) -> dict:
    return {"mean": round(e.mean, 5), "lo": round(e.lo, 5), "hi": round(e.hi, 5)}


def forecast_skill(preds: pd.DataFrame) -> dict:
    """Out-of-sample CRPS / MAE / interval coverage vs observed CLI highs, all stations."""
    p = preds.dropna(subset=["obs"])
    y = p["obs"].to_numpy(float)
    members = p[dataset.MEMBERS].to_numpy(float)
    raw_mu, raw_sd = members.mean(axis=1), members.std(axis=1, ddof=1) + 1e-6
    crps_emos = crps_normal(p["mu"].to_numpy(), p["sigma"].to_numpy(), y)
    crps_raw = crps_normal(raw_mu, raw_sd, y)
    dates = p["date"].to_numpy()
    pit_emos = norm.cdf(y, p["mu"], p["sigma"])
    return {
        "n_station_days": int(len(p)),
        "n_stations": int(p["series"].nunique()),
        "period": [str(p["date"].min()), str(p["date"].max())],
        "crps_emos": _est(cluster_bootstrap(crps_emos, dates)),
        "crps_raw_gaussian": _est(cluster_bootstrap(crps_raw, dates)),
        "crps_diff_emos_minus_raw": _est(cluster_bootstrap(crps_emos - crps_raw, dates)),
        "mae_emos_mean": _est(cluster_bootstrap(np.abs(y - p["mu"].to_numpy()), dates)),
        "mae_raw_mean": _est(cluster_bootstrap(np.abs(y - raw_mu), dates)),
        "coverage_80_emos": float(np.mean((pit_emos > 0.1) & (pit_emos < 0.9))),
        "coverage_80_raw": float(np.mean(np.abs(y - raw_mu) < norm.ppf(0.9) * raw_sd)),
    }, {"EMOS": pit_emos, "Raw Gaussian (mean ± sd of models)": norm.cdf(y, raw_mu, raw_sd)}


def market_comparison(br: pd.DataFrame) -> tuple[dict, list[dict]]:
    scores = {f: event_scores(br, f) for f in FORECASTERS}
    dates = scores["p_emos"]["date"].to_numpy()
    out: dict = {"n_events": int(len(scores["p_emos"])), "n_brackets": int(len(br)),
                 "n_days": int(br["date"].nunique()),
                 "period": [str(br["date"].min()), str(br["date"].max())],
                 "series": sorted(br["series"].unique().tolist())}
    rows = []
    for metric in ("brier", "log_loss", "rps"):
        out[metric] = {}
        for f in FORECASTERS:
            est = cluster_bootstrap(scores[f][metric].to_numpy(), dates)
            out[metric][f] = _est(est)
            rows.append({"forecaster": f, "metric": metric, **asdict(est)})
        diff = scores["p_emos"][metric].to_numpy() - scores["p_market"][metric].to_numpy()
        out[metric]["emos_minus_market"] = _est(cluster_bootstrap(diff, dates))
    # Per-city log-loss difference (negative = model better than market).
    per_city = {}
    se = scores["p_emos"].merge(scores["p_market"], on=["event_ticker", "date"],
                                suffixes=("_m", "_k"))
    se["series"] = se["event_ticker"].str.split("-").str[0]
    for s, g in se.groupby("series"):
        d = (g["log_loss_m"] - g["log_loss_k"]).to_numpy()
        per_city[s] = {"n": int(len(g)), **_est(cluster_bootstrap(d, g["date"].to_numpy()))}
    out["log_loss_emos_minus_market_by_series"] = per_city
    return out, rows


ABLATION_SERIES = ["KXHIGHAUS", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHNY"]


def fresh_run_ablation(lead: int) -> dict:
    """Does fresher model information close the gap to the market?

    On identical rows (4 long-history cities, dates with a decision-day ECMWF run),
    compare EMOS on the five fixed-lead models with EMOS that also sees the 00Z
    ECMWF HRES run of the decision day. Both are walk-forward, same spec.
    """
    base = dataset.emos_predictions(lead, ABLATION_SERIES, fresh_table=True)
    fresh = dataset.emos_predictions(lead, ABLATION_SERIES, fresh_table=True,
                                     members=[*dataset.MEMBERS, dataset.FRESH_MEMBER])
    out: dict = {"series": ABLATION_SERIES}
    for name, preds in (("base", base), ("fresh", fresh)):
        p = preds.dropna(subset=["obs"])
        crps = crps_normal(p["mu"].to_numpy(), p["sigma"].to_numpy(), p["obs"].to_numpy(float))
        out[f"crps_{name}"] = _est(cluster_bootstrap(crps, p["date"].to_numpy()))
        out[f"mae_{name}"] = round(float(np.abs(p["obs"] - p["mu"]).mean()), 4)
    bb = dataset.bracket_table(lead, base)
    bf = dataset.bracket_table(lead, fresh)
    bb = bb[bb["series"].isin(ABLATION_SERIES)]
    bf = bf[bf["series"].isin(ABLATION_SERIES)]
    merged = bf.merge(bb[["ticker", "p_emos"]].rename(columns={"p_emos": "p_base"}), on="ticker")
    scores = {c: event_scores(merged, c) for c in ("p_base", "p_emos", "p_market")}
    dates = scores["p_emos"]["date"].to_numpy()
    out["n_events"] = int(len(scores["p_emos"]))
    for metric in ("brier", "log_loss"):
        out[metric] = {
            "base": _est(cluster_bootstrap(scores["p_base"][metric].to_numpy(), dates)),
            "fresh": _est(cluster_bootstrap(scores["p_emos"][metric].to_numpy(), dates)),
            "market": _est(cluster_bootstrap(scores["p_market"][metric].to_numpy(), dates)),
            "fresh_minus_base": _est(cluster_bootstrap(
                scores["p_emos"][metric].to_numpy() - scores["p_base"][metric].to_numpy(), dates)),
            "fresh_minus_market": _est(cluster_bootstrap(
                scores["p_emos"][metric].to_numpy() - scores["p_market"][metric].to_numpy(),
                dates)),
        }
    trades = simulate(merged, BacktestConfig())
    summary = summarise(trades, BacktestConfig.contracts)
    out["backtest_margin_0.03"] = None if summary is None else asdict(summary)
    return out


def settlement_check() -> dict:
    """How often Kalshi's settled value equals the NWS CLI high, by settlement source."""
    from weather_edge.collect import RAW_DIR
    from weather_edge.stations import STATIONS

    rows = []
    for s, st in STATIONS.items():
        mp = RAW_DIR / "markets" / f"{s}.parquet"
        cp = RAW_DIR / "cli" / f"{st.icao}.parquet"
        if not (mp.exists() and cp.exists()):
            continue
        m = pd.read_parquet(mp).drop_duplicates("event_ticker")
        c = pd.read_parquet(cp)
        j = m.merge(c, on="date").dropna(subset=["expiration_value"])
        rows.append(j[["source", "expiration_value", "cli_high_f"]])
    j = pd.concat(rows)
    out = {}
    for src, g in j.groupby("source"):
        diff = g["expiration_value"] - g["cli_high_f"]
        out[src] = {"n_events": int(len(g)), "exact_match": float((diff == 0).mean()),
                    "within_1F": float((diff.abs() <= 1).mean()),
                    "mean_diff_F": float(diff.mean())}
    return out


def run(leads: tuple[int, ...] = LEADS) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                     "settlement_check": settlement_check(), "leads": {}}
    score_rows, cal_frames, trades_by_lead = [], {}, {}
    for lead in leads:
        preds = dataset.emos_predictions(lead)
        dataset.save(preds, f"emos_lead{lead}")
        skill, pits = forecast_skill(preds)
        if lead == 1:
            plots.pit_histograms(pits, FIGURES_DIR / "pit.png")
        br = dataset.bracket_table(lead, preds)
        dataset.save(br, f"brackets_lead{lead}")
        comp, rows = market_comparison(br)
        comp["dropped_events"] = br.attrs.get("dropped", {})
        score_rows += [{**r, "lead": lead} for r in rows]
        cal_frames[lead] = br

        backtests = {}
        for margin in MARGINS:
            cfg = BacktestConfig(margin=margin)
            trades = simulate(br, cfg)
            summary = summarise(trades, cfg.contracts)
            backtests[f"margin_{margin:.2f}"] = None if summary is None else asdict(summary)
            if margin == BacktestConfig.margin:
                trades_by_lead[lead] = trades
                if not trades.empty:
                    trades.to_csv(REPORTS_DIR / f"trades_lead{lead}.csv", index=False)
        results["leads"][str(lead)] = {"forecast_skill": skill, "vs_market": comp,
                                       "backtest": backtests,
                                       "fresh_run_ablation": fresh_run_ablation(lead)}

    plots.reliability(cal_frames, FIGURES_DIR / "reliability.png")
    plots.score_intervals(pd.DataFrame(score_rows), FIGURES_DIR / "scores.png")
    plots.cumulative_pnl(trades_by_lead, FIGURES_DIR / "pnl.png")
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results
