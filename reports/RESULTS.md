# Results

Generated 2026-10-02T01:44:36+00:00 by `weather-edge evaluate`. Intervals are 95% bootstrap CIs resampling dates.

## Settlement value vs NWS CLI

| Settlement source | Events | Exact match | Within 1°F |
|---|---|---|---|
| NWS_CLI | 5,950 | 99.88% | 100.00% |
| TWC | 1,108 | 99.91% | 99.91% |

## Lead 1

### Forecast skill vs observed CLI high (18,766 station-days, 24 stations, 2024-08-01 to 2026-09-30)

| | EMOS | Raw multi-model Gaussian |
|---|---|---|
| CRPS (°F) | 1.261 [1.238, 1.283] | 1.821 [1.794, 1.850] |
| MAE of mean (°F) | 1.730 [1.700, 1.759] | 2.512 [2.474, 2.551] |
| 80% interval coverage | 76.7% | 63.7% |

### Probabilities vs market (7,658 events, 24 series, 2024-08-01 to 2026-09-30)

| Forecaster | Brier | Log loss | RPS |
|---|---|---|---|
| EMOS model | 0.697 [0.691, 0.704] | 1.380 [1.361, 1.399] | 0.538 [0.527, 0.550] |
| Raw ensemble | 0.902 [0.891, 0.913] | 2.338 [2.290, 2.386] | 0.779 [0.764, 0.796] |
| Market (mid) | 0.560 [0.552, 0.569] | 1.026 [1.010, 1.043] | 0.357 [0.349, 0.365] |
| **EMOS − market** | 0.137 [0.128, 0.145] | 0.354 [0.335, 0.372] | 0.181 [0.171, 0.191] |

Events excluded: 683 with missing, stale (>3 h) or one-sided quotes; events before the first out-of-sample month have no forecast.

Log loss, EMOS − market, by series (negative = model better):

| Series | Events | Δ log loss |
|---|---|---|
| KXHIGHAUS | 713 | 0.376 [0.309, 0.438] |
| KXHIGHCHI | 669 | 0.410 [0.336, 0.478] |
| KXHIGHDEN | 640 | 0.382 [0.317, 0.447] |
| KXHIGHLAX | 580 | 0.468 [0.388, 0.544] |
| KXHIGHMIA | 649 | 0.278 [0.219, 0.340] |
| KXHIGHNY | 667 | 0.325 [0.261, 0.392] |
| KXHIGHPHIL | 605 | 0.392 [0.330, 0.463] |
| KXHIGHTATL | 229 | 0.362 [0.266, 0.462] |
| KXHIGHTBOS | 230 | 0.341 [0.245, 0.434] |
| KXHIGHTDAL | 224 | 0.381 [0.286, 0.476] |
| KXHIGHTDC | 249 | 0.425 [0.324, 0.528] |
| KXHIGHTEWR | 20 | -0.036 [-0.365, 0.282] |
| KXHIGHTHOU | 224 | 0.373 [0.278, 0.470] |
| KXHIGHTLV | 248 | 0.189 [0.111, 0.259] |
| KXHIGHTMIN | 232 | 0.307 [0.194, 0.412] |
| KXHIGHTNOLA | 251 | 0.353 [0.256, 0.450] |
| KXHIGHTOKC | 220 | 0.433 [0.327, 0.543] |
| KXHIGHTPHX | 228 | 0.204 [0.109, 0.296] |
| KXHIGHTSAN | 21 | 0.153 [-0.139, 0.455] |
| KXHIGHTSATX | 222 | 0.413 [0.321, 0.508] |
| KXHIGHTSDF | 13 | 0.611 [0.449, 0.790] |
| KXHIGHTSEA | 253 | 0.215 [0.131, 0.299] |
| KXHIGHTSFO | 251 | 0.284 [0.188, 0.380] |
| KXHIGHTTTN | 20 | 0.211 [-0.092, 0.599] |

### Backtest (10 contracts per signal, taker, fees and spread included)

| Margin | Trades | Days | Fees | P&L | ROI on capital | P&L per trade |
|---|---|---|---|---|---|---|
| 0.00 | 33,179 | 758 | $2,881 | −$5,664 | -4.8% [-5.9%, -3.7%] | -0.171 [-0.208, -0.132] |
| 0.02 | 26,631 | 756 | $2,555 | −$5,065 | -5.7% [-7.1%, -4.4%] | -0.190 [-0.239, -0.148] |
| 0.03 | 24,465 | 755 | $2,413 | −$4,820 | -6.0% [-7.4%, -4.7%] | -0.197 [-0.241, -0.149] |
| 0.05 | 21,041 | 753 | $2,149 | −$4,035 | -5.9% [-7.5%, -4.4%] | -0.192 [-0.243, -0.138] |
| 0.10 | 14,510 | 747 | $1,559 | −$2,586 | -5.7% [-7.8%, -3.5%] | -0.178 [-0.246, -0.111] |

### Ablation: add the decision-day 00Z ECMWF run (KXHIGHAUS, KXHIGHCHI, KXHIGHMIA, KXHIGHNY; 2,595 events)

| | Base EMOS | + fresh run | Market | Fresh − base | Fresh − market |
|---|---|---|---|---|---|
| CRPS (°F) | 1.160 [1.115, 1.206] | 1.137 [1.096, 1.181] | | | |
| Brier | 0.690 [0.678, 0.702] | 0.683 [0.671, 0.696] | 0.551 [0.537, 0.564] | -0.006 [-0.010, -0.003] | 0.133 [0.118, 0.147] |
| Log Loss | 1.366 [1.330, 1.403] | 1.347 [1.311, 1.385] | 1.000 [0.974, 1.026] | -0.019 [-0.029, -0.009] | 0.347 [0.310, 0.384] |

Backtest with the fresh model (margin 0.03): 7,767 trades, P&L −$983, ROI -3.7% [-6.3%, -1.1%].

## Lead 2

### Forecast skill vs observed CLI high (18,932 station-days, 24 stations, 2024-08-01 to 2026-09-30)

| | EMOS | Raw multi-model Gaussian |
|---|---|---|
| CRPS (°F) | 1.487 [1.459, 1.515] | 2.005 [1.967, 2.040] |
| MAE of mean (°F) | 2.041 [2.004, 2.079] | 2.740 [2.693, 2.787] |
| 80% interval coverage | 76.7% | 63.4% |

### Probabilities vs market (7,193 events, 24 series, 2024-08-01 to 2026-09-30)

| Forecaster | Brier | Log loss | RPS |
|---|---|---|---|
| EMOS model | 0.749 [0.742, 0.756] | 1.533 [1.513, 1.554] | 0.643 [0.629, 0.656] |
| Raw ensemble | 0.922 [0.911, 0.933] | 2.459 [2.410, 2.508] | 0.829 [0.811, 0.846] |
| Market (mid) | 0.688 [0.682, 0.694] | 1.357 [1.342, 1.373] | 0.523 [0.512, 0.533] |
| **EMOS − market** | 0.061 [0.055, 0.067] | 0.176 [0.157, 0.194] | 0.120 [0.109, 0.130] |

Events excluded: 1,290 with missing, stale (>3 h) or one-sided quotes; events before the first out-of-sample month have no forecast.

Log loss, EMOS − market, by series (negative = model better):

| Series | Events | Δ log loss |
|---|---|---|
| KXHIGHAUS | 713 | 0.256 [0.189, 0.325] |
| KXHIGHCHI | 723 | 0.176 [0.117, 0.235] |
| KXHIGHDEN | 641 | 0.166 [0.108, 0.221] |
| KXHIGHLAX | 588 | 0.319 [0.251, 0.384] |
| KXHIGHMIA | 504 | 0.029 [-0.035, 0.098] |
| KXHIGHNY | 511 | 0.125 [0.065, 0.184] |
| KXHIGHPHIL | 433 | 0.096 [0.033, 0.159] |
| KXHIGHTATL | 204 | 0.234 [0.139, 0.330] |
| KXHIGHTBOS | 203 | 0.144 [0.057, 0.237] |
| KXHIGHTDAL | 228 | 0.162 [0.082, 0.249] |
| KXHIGHTDC | 204 | 0.207 [0.113, 0.300] |
| KXHIGHTEWR | 17 | -0.068 [-0.409, 0.260] |
| KXHIGHTHOU | 229 | 0.207 [0.126, 0.291] |
| KXHIGHTLV | 256 | 0.060 [-0.012, 0.134] |
| KXHIGHTMIN | 235 | 0.202 [0.124, 0.288] |
| KXHIGHTNOLA | 253 | 0.200 [0.105, 0.294] |
| KXHIGHTOKC | 229 | 0.260 [0.186, 0.336] |
| KXHIGHTPHX | 235 | 0.147 [0.049, 0.243] |
| KXHIGHTSAN | 15 | -0.103 [-0.310, 0.151] |
| KXHIGHTSATX | 228 | 0.220 [0.137, 0.307] |
| KXHIGHTSDF | 16 | 0.310 [-0.063, 0.678] |
| KXHIGHTSEA | 257 | 0.172 [0.096, 0.246] |
| KXHIGHTSFO | 256 | 0.132 [0.047, 0.213] |
| KXHIGHTTTN | 15 | -0.062 [-0.204, 0.074] |

### Backtest (10 contracts per signal, taker, fees and spread included)

| Margin | Trades | Days | Fees | P&L | ROI on capital | P&L per trade |
|---|---|---|---|---|---|---|
| 0.00 | 32,676 | 742 | $3,039 | −$3,136 | -1.8% [-2.7%, -1.0%] | -0.096 [-0.140, -0.050] |
| 0.02 | 24,561 | 740 | $2,538 | −$1,772 | -1.5% [-2.7%, -0.4%] | -0.072 [-0.126, -0.019] |
| 0.03 | 21,684 | 740 | $2,306 | −$1,640 | -1.6% [-3.0%, -0.5%] | -0.076 [-0.130, -0.020] |
| 0.05 | 17,266 | 739 | $1,895 | −$1,193 | -1.6% [-3.2%, -0.1%] | -0.069 [-0.134, -0.004] |
| 0.10 | 9,488 | 729 | $1,092 | −$442 | -1.1% [-3.4%, +1.1%] | -0.047 [-0.135, +0.043] |

### Ablation: add the decision-day 00Z ECMWF run (KXHIGHAUS, KXHIGHCHI, KXHIGHMIA, KXHIGHNY; 2,303 events)

| | Base EMOS | + fresh run | Market | Fresh − base | Fresh − market |
|---|---|---|---|---|---|
| CRPS (°F) | 1.374 [1.323, 1.429] | 1.362 [1.312, 1.416] | | | |
| Brier | 0.741 [0.728, 0.754] | 0.738 [0.725, 0.750] | 0.686 [0.676, 0.696] | -0.003 [-0.006, -0.001] | 0.052 [0.040, 0.064] |
| Log Loss | 1.524 [1.484, 1.562] | 1.512 [1.472, 1.551] | 1.351 [1.325, 1.376] | -0.012 [-0.021, -0.003] | 0.162 [0.127, 0.197] |

Backtest with the fresh model (margin 0.03): 6,689 trades, P&L $625, ROI +1.9% [-0.1%, +4.1%].
