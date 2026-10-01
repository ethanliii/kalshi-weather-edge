"""Turn a temperature forecast into probabilities for a market's brackets.

The settlement value is an integer °F (the CLI rounds the observed maximum).
Treating the latent maximum T as continuous, the reported value is k when
k - 0.5 <= T < k + 0.5, so a bracket [lo, hi] (inclusive integers) has

    P(bracket) = F(hi + 0.5) - F(lo - 0.5)

with F the forecast CDF and infinite edges for the tail brackets. Because the
brackets tile the integers, the edges are shared between neighbours and the
probabilities telescope to F(+inf) - F(-inf) = 1 exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from weather_edge.kalshi.markets import Bracket, validate_partition


def _edges(brackets: Sequence[Bracket]) -> np.ndarray:
    validate_partition(list(brackets))
    inner = [b.lo - 0.5 for b in brackets[1:]]
    return np.array([-np.inf, *inner, np.inf])


def probs_from_cdf(cdf, brackets: Sequence[Bracket]) -> np.ndarray:
    """Bracket probabilities from any CDF; brackets must be sorted and a partition."""
    cdf_vals = np.asarray(cdf(_edges(brackets)), dtype=float)
    cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
    p = np.diff(cdf_vals)
    return np.clip(p, 0.0, 1.0) / np.clip(p, 0.0, 1.0).sum()


def probs_normal(mu: float, sigma: float, brackets: Sequence[Bracket]) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return probs_from_cdf(lambda x: norm.cdf(x, loc=mu, scale=sigma), brackets)


def probs_members(members: Sequence[float], brackets: Sequence[Bracket]) -> np.ndarray:
    """Raw ensemble probabilities: share of members whose rounded value lands in each bracket.

    This is the uncalibrated baseline: with few members it assigns exactly zero to
    many brackets, which is why statistical post-processing is needed.
    """
    m = np.asarray([x for x in members if np.isfinite(x)], dtype=float)
    if m.size == 0:
        raise ValueError("no finite members")
    reported = np.floor(m + 0.5)  # round half up, like the CLI's integer report
    edges = _edges(brackets)
    counts = np.histogram(reported, bins=edges)[0]
    return counts / m.size
