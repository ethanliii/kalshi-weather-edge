"""Static figures for the README (matplotlib, PNG).

Colour roles follow a validated categorical palette (first three slots pass
colour-vision-deficiency checks for every pair): EMOS = blue, market = orange,
raw ensemble = aqua. Text stays in neutral ink; grids are recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from weather_edge.evaluation.metrics import calibration_table  # noqa: E402

COLORS = {"p_emos": "#2a78d6", "p_market": "#eb6834", "p_raw": "#1baf7a"}
LABELS = {"p_emos": "EMOS model", "p_market": "Market (mid)", "p_raw": "Raw ensemble"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LEAD_NAMES = {1: "Decision 10:00 LST on the day (lead 1)",
              2: "Decision 10:00 LST the day before (lead 2)"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _fig(ncols: int, width: float = 5.2, height: float = 4.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes.flat


def reliability(tables: dict[int, pd.DataFrame], path: Path) -> None:
    """Reliability diagram per lead: predicted probability vs observed frequency."""
    fig, axes = _fig(len(tables))
    for ax, (lead, df) in zip(axes, sorted(tables.items()), strict=True):
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--",
                label="Perfect calibration")
        for col in ("p_emos", "p_market", "p_raw"):
            cal = calibration_table(df[col].to_numpy(), df["y"].to_numpy(), bins=10)
            cal = cal[cal["n"] >= 20]
            ax.errorbar(cal["mean_pred"], cal["obs_freq"],
                        yerr=[cal["obs_freq"] - cal["ci_lo"], cal["ci_hi"] - cal["obs_freq"]],
                        color=COLORS[col], linewidth=2, marker="o", markersize=5, capsize=0,
                        elinewidth=1, label=LABELS[col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability (bracket settles YES)")
        ax.set_ylabel("Observed frequency")
        ax.set_title(LEAD_NAMES[lead], fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle("Calibration of bracket probabilities (bins with n ≥ 20, 95% Wilson CIs)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def score_intervals(scores: pd.DataFrame, path: Path) -> None:
    """Dot-and-whisker of mean log loss and Brier per forecaster with bootstrap CIs.

    `scores` columns: lead, forecaster, metric, mean, lo, hi.
    """
    metrics = ["log_loss", "brier"]
    titles = {"log_loss": "Log loss per event (lower is better)",
              "brier": "Brier score per event (lower is better)"}
    fig, axes = _fig(2, width=5.2, height=3.6)
    order = ["p_emos", "p_raw", "p_market"]
    for ax, metric in zip(axes, metrics, strict=True):
        sub = scores[scores["metric"] == metric]
        yticks, ylabels = [], []
        for i, lead in enumerate(sorted(sub["lead"].unique())):
            for j, f in enumerate(order):
                r = sub[(sub["lead"] == lead) & (sub["forecaster"] == f)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i * (len(order) + 1) + j
                ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                            color=COLORS[f], marker="o", markersize=7, linewidth=2, capsize=0)
                yticks.append(y)
                ylabels.append(f"{LABELS[f]} · lead {lead}")
        ax.set_yticks(yticks, ylabels)
        ax.invert_yaxis()
        ax.set_title(titles[metric], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def cumulative_pnl(trades: dict[int, pd.DataFrame], path: Path) -> None:
    fig, axes = _fig(1, width=8, height=3.8)
    ax = axes[0]
    lead_colors = {1: "#2a78d6", 2: "#eb6834"}
    for lead, t in sorted(trades.items()):
        if t.empty:
            continue
        daily = t.groupby("date")["pnl"].sum().sort_index()
        cum = daily.cumsum()
        ax.plot(pd.to_datetime(cum.index), cum.values, color=lead_colors[lead], linewidth=2,
                label=f"Lead {lead} ({len(t)} trades)")
        ax.annotate(f"${cum.iloc[-1]:,.0f}", (pd.to_datetime(cum.index[-1]), cum.iloc[-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=8, color=INK2,
                    va="center")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("Cumulative P&L after fees ($)")
    ax.set_title("Backtest: 10-contract taker orders when model edge > fees + 3¢", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pit_histograms(pits: dict[str, np.ndarray], path: Path) -> None:
    """PIT histograms: flat = calibrated, U-shape = overconfident, hump = underconfident."""
    fig, axes = _fig(len(pits), width=4.2, height=3.4)
    colors = {"EMOS": "#2a78d6", "Raw Gaussian (mean ± sd of models)": "#1baf7a"}
    for ax, (name, pit) in zip(axes, pits.items(), strict=True):
        ax.hist(pit, bins=20, range=(0, 1), density=True, color=colors.get(name, "#2a78d6"),
                edgecolor=SURFACE, linewidth=2)
        ax.axhline(1, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("PIT value")
    axes[0].set_ylabel("Density")
    fig.suptitle("Probability integral transform of observed highs (out of sample, lead 1)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
