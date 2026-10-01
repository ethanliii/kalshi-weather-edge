from weather_edge.evaluation.render import render

E = {"mean": 1.0, "lo": 0.9, "hi": 1.1}
NEG = {"mean": -0.05, "lo": -0.08, "hi": -0.02}


def results():
    bt = {"n_trades": 12, "n_days": 5, "total_pnl": -3.5, "capital_deployed": 40.0, "fees_paid": 1.2,
          "hit_rate": 0.4, "pnl_per_trade": NEG, "roi": NEG, "mean_edge": 0.1, "realised_edge": -0.03}
    lead = {
        "forecast_skill": {"n_station_days": 100, "n_stations": 2, "period": ["2025-01-01", "2025-03-01"],
                           "crps_emos": E, "crps_raw_gaussian": E, "mae_emos_mean": E, "mae_raw_mean": E,
                           "coverage_80_emos": 0.78, "coverage_80_raw": 0.6},
        "vs_market": {"n_events": 50, "series": ["KXHIGHNY"], "period": ["2025-01-01", "2025-03-01"],
                      **{m: {"p_emos": E, "p_raw": E, "p_market": E, "emos_minus_market": NEG}
                         for m in ("brier", "log_loss", "rps")},
                      "dropped_events": {"bad_quotes": 3},
                      "log_loss_emos_minus_market_by_series": {"KXHIGHNY": {"n": 50, **NEG}}},
        "backtest": {"margin_0.03": bt, "margin_0.10": None},
        "fresh_run_ablation": {"series": ["KXHIGHNY"], "n_events": 40, "crps_base": E, "crps_fresh": E,
                               **{m: {"base": E, "fresh": E, "market": E, "fresh_minus_base": NEG,
                                      "fresh_minus_market": NEG} for m in ("brier", "log_loss")},
                               "backtest_margin_0.03": bt},
    }
    return {"generated_utc": "2026-10-01T00:00:00+00:00",
            "settlement_check": {"NWS_CLI": {"n_events": 10, "exact_match": 1.0, "within_1F": 1.0}},
            "leads": {"1": lead}}


def test_render_contains_every_section_and_formats_money():
    md = render(results())
    for heading in ("## Settlement value vs NWS CLI", "## Lead 1", "### Backtest", "### Ablation"):
        assert heading in md
    assert "−$4" in md  # negative P&L rendered with a minus sign, rounded
    assert "| 0.10 | 0 |" in md  # margin with no trades still listed
    assert "{{" not in md
