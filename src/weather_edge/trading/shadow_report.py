"""Score shadow-mode orders against actual settlements.

Each `shadow_order` in the ledger is an IOC limit order the bot would have sent
at the then-current best price, capped by displayed size. Assuming it would
have filled in full at that price, P&L = count * 1{won} - count * price - fee.
Unsettled markets are reported as pending.
"""

from __future__ import annotations

import pandas as pd

from weather_edge.kalshi.client import KalshiClient
from weather_edge.trading.risk import Ledger


def score(ledger: Ledger, client: KalshiClient | None = None) -> pd.DataFrame:
    rows = ledger.records("shadow_order")
    if not rows:
        return pd.DataFrame()
    client = client or KalshiClient()
    results: dict[str, str | None] = {}
    for t in {r["ticker"] for r in rows}:
        m = client.get_market(t)
        results[t] = m.get("result") if m.get("result") in ("yes", "no") else None
    out = []
    for r in rows:
        result = results[r["ticker"]]
        won = None if result is None else (result == r["side"])
        pnl = None if won is None else r["count"] * float(won) - r["cost"]
        out.append({"ts": r["ts"], "ticker": r["ticker"], "side": r["side"],
                    "price": r["price"], "count": r["count"], "prob": r["prob"],
                    "edge": r["edge"], "fee": r["fee"], "cost": r["cost"],
                    "result": result, "won": won, "pnl": pnl})
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No shadow orders logged yet."
    settled = df.dropna(subset=["pnl"])
    lines = [f"shadow orders: {len(df)} ({len(settled)} settled, {len(df) - len(settled)} pending)",
             f"capital that would have been deployed: ${df['cost'].sum():,.2f}"]
    if not settled.empty:
        pnl, cost = settled["pnl"].sum(), settled["cost"].sum()
        lines += [f"settled P&L after fees: {'-' if pnl < 0 else ''}${abs(pnl):,.2f} "
                  f"(ROI {pnl / cost:+.1%} on ${cost:,.2f})",
                  f"hit rate {settled['won'].mean():.1%}; mean expected edge "
                  f"{settled['edge'].mean() * 100:.1f}c vs realised "
                  f"{settled['pnl'].sum() / settled['count'].sum() * 100:+.1f}c per contract"]
    return "\n".join(lines)
