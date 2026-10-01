# Forward test (written by `.github/workflows/daily.yml`)

| Path | Contents |
|---|---|
| `snapshots/YYYY-MM-DD.csv` | EMOS / raw-ensemble / market-implied probability for every bracket of each open event, taken within 2 h after its 10:00 LST decision time. Production quotes, read-only. |
| `ensemble/YYYY-MM-DD.csv.gz` | GEFS (31) + ECMWF ENS (51) member CLI-day highs. Open-Meteo keeps only ~3 months of ensemble history, so this archive is what makes a true-ensemble model evaluable later. |
| `forward_scores.csv` | Per-event Brier and log loss for model, raw ensemble and market once the event settles. |

These are genuinely out-of-sample: every row is written before the outcome exists.
