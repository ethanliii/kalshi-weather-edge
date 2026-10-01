"""Historical backtest of a simple taker strategy against decision-time quotes.

At each event's decision time the strategy looks at every bracket:

  * buy YES at the ask if  p_model - ask - fee      > margin
  * buy NO  at 1 - bid if  (1 - p_model) - (1 - bid) - fee > margin

where the fee is Kalshi's taker fee for an order of `contracts` contracts. Orders
cross the spread (we pay the ask / hit the bid), so the spread is fully charged.
Each signal trades a flat `contracts` contracts and is held to settlement.

Assumptions (also listed in the README):
  * fills at the top-of-book quote from the hourly candle that closed at or before
    the decision time; book depth is unknown historically, so size is kept small;
  * no market impact and no queue position effects;
  * the margin is fixed before looking at results; other margins are reported as
    a sensitivity table, not selected on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from weather_edge.evaluation.metrics import Estimate, cluster_bootstrap
from weather_edge.trading.fees import order_fee
from weather_edge.trading.sizing import SizingConfig, best_opportunity


@dataclass(frozen=True)
class BacktestConfig:
    margin: float = 0.03
    contracts: int = 10
    prob_col: str = "p_emos"


def simulate(brackets: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """One row per trade with its P&L in dollars."""
    sizing = SizingConfig(min_edge=cfg.margin)
    trades = []
    for row in brackets.itertuples(index=False):
        p = getattr(row, cfg.prob_col)
        opp = best_opportunity(p, row.yes_bid, row.yes_ask, sizing, ref_count=cfg.contracts)
        if opp is None:
            continue
        won = (row.y == 1) if opp.side == "yes" else (row.y == 0)
        fee = order_fee(opp.price, cfg.contracts)
        cost = opp.price * cfg.contracts + fee
        pnl = (cfg.contracts if won else 0.0) - cost
        trades.append({
            "date": row.date, "series": row.series, "event_ticker": row.event_ticker,
            "ticker": row.ticker, "side": opp.side, "price": opp.price, "prob": opp.prob,
            "edge": opp.edge, "fee": fee, "cost": cost, "won": won, "pnl": pnl,
        })
    return pd.DataFrame(trades)


@dataclass(frozen=True)
class BacktestSummary:
    n_trades: int
    n_days: int
    total_pnl: float
    capital_deployed: float
    fees_paid: float
    hit_rate: float
    pnl_per_trade: Estimate  # dollars, date-clustered bootstrap CI
    roi: Estimate  # pnl / capital deployed, date-clustered bootstrap CI
    mean_edge: float  # model's expected edge per contract at entry
    realised_edge: float  # realised pnl per contract


def summarise(trades: pd.DataFrame, contracts: int, seed: int = 0) -> BacktestSummary | None:
    if trades.empty:
        return None
    dates = trades["date"].to_numpy()
    per_trade = cluster_bootstrap(trades["pnl"].to_numpy(), dates, seed=seed)
    # ROI is a ratio of sums, so bootstrap it directly on resampled dates.
    daily = trades.groupby("date")[["pnl", "cost"]].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(2000, len(daily)))
    pnl, cost = daily["pnl"].to_numpy(), daily["cost"].to_numpy()
    reps = pnl[idx].sum(axis=1) / cost[idx].sum(axis=1)
    roi = Estimate(pnl.sum() / cost.sum(), *np.quantile(reps, [0.025, 0.975]))
    return BacktestSummary(
        n_trades=len(trades),
        n_days=trades["date"].nunique(),
        total_pnl=float(trades["pnl"].sum()),
        capital_deployed=float(trades["cost"].sum()),
        fees_paid=float(trades["fee"].sum()),
        hit_rate=float(trades["won"].mean()),
        pnl_per_trade=per_trade,
        roi=roi,
        mean_edge=float(trades["edge"].mean()),
        realised_edge=float(trades["pnl"].sum() / (len(trades) * contracts)),
    )
