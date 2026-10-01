import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from weather_edge.dataset import _event_ok, market_implied
from weather_edge.evaluation.backtest import BacktestConfig, simulate, summarise
from weather_edge.evaluation.metrics import (
    EPS,
    calibration_table,
    cluster_bootstrap,
    event_scores,
)


def event(ev, d, probs, winner, bids=None, asks=None):
    n = len(probs)
    lo = [-math.inf] + [70 + 2 * i for i in range(n - 1)]
    hi = [69 + 2 * i for i in range(n - 1)] + [math.inf]
    return pd.DataFrame({
        "event_ticker": ev, "date": d, "series": "KXHIGHNY", "ticker": [f"{ev}-{i}" for i in range(n)],
        "lo": lo, "hi": hi, "p": probs, "y": [int(i == winner) for i in range(n)],
        "result": ["yes" if i == winner else "no" for i in range(n)],
        "yes_bid": bids or [0.0] * n, "yes_ask": asks or [0.01] * n,
        "decision_time": pd.Timestamp("2026-01-01T15:00", tz="UTC"),
        "candle_end": pd.Timestamp("2026-01-01T15:00", tz="UTC"),
    })


def test_event_scores_known_values():
    df = event("E1", date(2026, 1, 1), [0.1, 0.6, 0.3], winner=1)
    s = event_scores(df, "p").iloc[0]
    assert s["brier"] == pytest.approx(0.1**2 + 0.4**2 + 0.3**2)
    assert s["log_loss"] == pytest.approx(-math.log(0.6))
    # RPS: cumulative p = [.1, .7, 1], cumulative y = [0, 1, 1]
    assert s["rps"] == pytest.approx(0.1**2 + 0.3**2)


def test_log_loss_clipped_for_zero_probability():
    df = event("E1", date(2026, 1, 1), [1.0, 0.0, 0.0], winner=2)
    assert event_scores(df, "p").iloc[0]["log_loss"] == pytest.approx(-math.log(EPS))


def test_perfect_forecast_scores_zero():
    df = event("E1", date(2026, 1, 1), [0.0, 1.0, 0.0], winner=1)
    s = event_scores(df, "p").iloc[0]
    assert s["brier"] == 0 and s["rps"] == 0 and s["log_loss"] == pytest.approx(0)


def test_cluster_bootstrap_ci_contains_mean_and_widens_with_clustering():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(50), 20)
    shared = rng.normal(0, 1, 50)[clusters]  # strong within-cluster correlation
    values = shared + rng.normal(0, 0.1, clusters.size)
    clustered = cluster_bootstrap(values, clusters)
    naive = cluster_bootstrap(values, np.arange(values.size))
    assert clustered.lo < clustered.mean < clustered.hi
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_calibration_table_perfectly_calibrated_data():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50_000)
    y = rng.uniform(0, 1, p.size) < p
    cal = calibration_table(p, y)
    assert len(cal) == 10
    assert ((cal["ci_lo"] <= cal["mean_pred"] + 0.01) & (cal["ci_hi"] >= cal["mean_pred"] - 0.01)).all()


def test_market_implied_sums_to_one():
    p = market_implied(np.array([0.10, 0.45, 0.40]), np.array([0.12, 0.48, 0.44]))
    assert p.sum() == pytest.approx(1.0)


def test_event_filter():
    good = event("E", date(2026, 1, 1), [0.2, 0.5, 0.3], 1, bids=[0.18, 0.48, 0.28], asks=[0.2, 0.5, 0.3])
    assert _event_ok(good)
    assert not _event_ok(good.assign(yes_ask=[0.2, 0.5, None]))  # missing quote
    assert not _event_ok(good.assign(result=["yes", "yes", "no"]))  # two winners
    assert not _event_ok(good.assign(yes_bid=0.0, yes_ask=0.01))  # mids sum far below 1
    stale = good.copy()
    stale.loc[0, "candle_end"] = pd.Timestamp("2025-12-31T15:00", tz="UTC")  # day-old quote
    assert not _event_ok(stale)


def test_backtest_pnl_accounting():
    # Model says bracket 1 is 80% likely; ask is 0.50 -> buy YES, and it wins.
    df = event("E", date(2026, 1, 1), [0.1, 0.8, 0.1], 1, bids=[0.05, 0.48, 0.40], asks=[0.08, 0.50, 0.45])
    df = df.rename(columns={"p": "p_emos"})
    trades = simulate(df, BacktestConfig(margin=0.03, contracts=10))
    yes = trades[trades["side"] == "yes"].iloc[0]
    assert yes["fee"] == pytest.approx(0.18)  # ceil(0.07*10*.5*.5 = 0.175)
    assert yes["pnl"] == pytest.approx(10 - 5.0 - 0.18)
    # Bracket 2: model 10% vs bid 0.40 -> buy NO at 0.60, wins because bracket 2 lost.
    no = trades[trades["ticker"] == "E-2"].iloc[0]
    assert no["side"] == "no" and no["price"] == pytest.approx(0.60) and no["won"]
    s = summarise(trades, 10)
    assert s.n_trades == len(trades) and s.total_pnl == pytest.approx(trades["pnl"].sum())


def test_backtest_no_trades_without_edge():
    df = event("E", date(2026, 1, 1), [0.2, 0.5, 0.3], 1, bids=[0.18, 0.48, 0.28], asks=[0.22, 0.52, 0.32])
    assert simulate(df.rename(columns={"p": "p_emos"}), BacktestConfig()).empty
    assert summarise(pd.DataFrame(), 10) is None
