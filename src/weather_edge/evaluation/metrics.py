"""Proper scoring rules, calibration, and cluster-bootstrap confidence intervals.

Scores are computed per *event* (one city-day, i.e. one categorical forecast
over its brackets):

  * Brier score (multi-category): sum_k (p_k - y_k)^2, in [0, 2]
  * Log loss: -log p_winner, with p clipped at EPS so a single
    zero-probability outcome doesn't make the average infinite
  * Ranked probability score: Brier on the cumulative distribution, which
    rewards putting mass *near* the outcome (brackets are ordered)

Uncertainty: bracket outcomes within an event are dependent, and events on the
same date are correlated across cities (one weather system hits several), so
the bootstrap resamples whole *dates*, not individual brackets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 0.005  # half a cent: below Kalshi's price granularity


def event_scores(df: pd.DataFrame, prob_col: str, event_col: str = "event_ticker") -> pd.DataFrame:
    """Per-event Brier, log loss and RPS. `df` has one row per bracket, sorted by bracket."""

    def one(g: pd.DataFrame) -> pd.Series:
        p = g[prob_col].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        return pd.Series({
            "brier": np.sum((p - y) ** 2),
            "log_loss": -np.log(np.clip(p[y == 1], EPS, 1)).sum(),
            "rps": np.sum((np.cumsum(p) - np.cumsum(y))[:-1] ** 2),
        })

    keyed = df.sort_values([event_col, "lo"])
    scores = keyed.groupby(event_col, sort=False)[[prob_col, "y"]].apply(one)
    dates = keyed.groupby(event_col, sort=False)["date"].first()
    return scores.assign(date=dates).reset_index()


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stat=np.mean,
) -> Estimate:
    """Percentile CI for `stat(values)` resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    codes, uniq = pd.factorize(np.asarray(clusters))
    n_clusters = len(uniq)
    rng = np.random.default_rng(seed)
    # Pre-sum per cluster so each replicate is a cheap weighted combination.
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(float)
    if stat is np.mean:
        draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        w = np.apply_along_axis(np.bincount, 1, draws, minlength=n_clusters)
        reps = (w @ sums) / (w @ counts)
    else:
        groups = [values[codes == i] for i in range(n_clusters)]
        reps = np.array([
            stat(np.concatenate([groups[j] for j in rng.integers(0, n_clusters, n_clusters)]))
            for _ in range(n_boot)
        ])
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(stat(values)), float(lo), float(hi))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data with Wilson 95% intervals for the observed frequency."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    z = 1.96
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        freq = y[mask].mean()
        denom = 1 + z**2 / n
        centre = (freq + z**2 / (2 * n)) / denom
        half = z * np.sqrt(freq * (1 - freq) / n + z**2 / (4 * n**2)) / denom
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": n,
                     "mean_pred": p[mask].mean(), "obs_freq": freq,
                     "ci_lo": centre - half, "ci_hi": centre + half})
    return pd.DataFrame(rows)


def skill_score(model: float, reference: float) -> float:
    """1 - model/reference: positive when the model beats the reference."""
    return 1 - model / reference
