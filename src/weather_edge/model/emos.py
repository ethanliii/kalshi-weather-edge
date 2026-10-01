"""Ensemble Model Output Statistics (EMOS) for daily maximum temperature.

Following Gneiting et al. (2005, Mon. Wea. Rev.), the predictive distribution is
Gaussian with a mean that is a weighted combination of the member forecasts and
a spread that grows with the ensemble spread:

    mu    = a0 + a1*sin(doy) + a2*cos(doy) + sum_k b_k * x_k
    log s = c0 + c1*log(S + 0.5) + c2*sin(doy) + c3*cos(doy)

where x_k are the individual model forecasts (non-exchangeable members: each
model gets its own weight), S is the standard deviation across models, and the
seasonal harmonics absorb the annual cycle in bias and predictability (marine
layer in summer, cold-air damming in winter, ...). The log link keeps the
spread positive without constrained optimisation.

Parameters minimise the mean continuous ranked probability score (CRPS), which
has a closed form for the normal distribution. CRPS is a proper scoring rule, so
minimising it rewards calibration and sharpness together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

_SQRT_PI = np.sqrt(np.pi)


def crps_normal(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at observation y."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / _SQRT_PI)


def seasonal(doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * np.asarray(doy, dtype=float) / 365.25
    return np.sin(angle), np.cos(angle)


@dataclass
class GaussianEMOS:
    members: list[str]
    params_: np.ndarray | None = field(default=None, repr=False)
    train_crps_: float | None = None

    # --- design ---------------------------------------------------------------
    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df[self.members].to_numpy(dtype=float)
        s, c = seasonal(pd.to_datetime(df["date"]).dt.dayofyear.to_numpy())
        loc_x = np.column_stack([np.ones(len(df)), s, c, x])
        spread = x.std(axis=1, ddof=1) if x.shape[1] > 1 else np.zeros(len(df))
        scale_x = np.column_stack([np.ones(len(df)), np.log(spread + 0.5), s, c])
        return loc_x, scale_x

    def _unpack(self, theta: np.ndarray, n_loc: int) -> tuple[np.ndarray, np.ndarray]:
        return theta[:n_loc], theta[n_loc:]

    # --- fit / predict ----------------------------------------------------------
    def fit(self, df: pd.DataFrame, y_col: str = "obs") -> GaussianEMOS:
        df = df.dropna(subset=[*self.members, y_col])
        if len(df) < 30:
            raise ValueError(f"need at least 30 training rows, got {len(df)}")
        loc_x, scale_x = self._design(df)
        y = df[y_col].to_numpy(dtype=float)

        # Start from least squares for the mean and the residual spread for the scale.
        beta0, *_ = np.linalg.lstsq(loc_x, y, rcond=None)
        resid_sd = np.std(y - loc_x @ beta0)
        gamma0 = np.array([np.log(max(resid_sd, 0.5)), 0.0, 0.0, 0.0])
        theta0 = np.concatenate([beta0, gamma0])
        n_loc = loc_x.shape[1]

        def objective(theta: np.ndarray) -> float:
            beta, gamma = self._unpack(theta, n_loc)
            sigma = np.exp(np.clip(scale_x @ gamma, -5, 5))
            return float(crps_normal(loc_x @ beta, sigma, y).mean())

        res = minimize(objective, theta0, method="L-BFGS-B")
        self.params_ = res.x
        self.train_crps_ = float(res.fun)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.params_ is None:
            raise RuntimeError("model is not fitted")
        loc_x, scale_x = self._design(df)
        beta, gamma = self._unpack(self.params_, loc_x.shape[1])
        return loc_x @ beta, np.exp(np.clip(scale_x @ gamma, -5, 5))


def walk_forward(
    df: pd.DataFrame,
    members: list[str],
    min_train_days: int = 180,
    refit: str = "MS",
    y_col: str = "obs",
) -> pd.DataFrame:
    """Out-of-sample EMOS predictions with an expanding window, refit monthly.

    For each month, the model is fitted only on rows dated strictly before the
    month's first day, so every prediction is genuinely out of sample.
    `df` must hold one station and one lead: columns date, members..., y_col.
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    first_test = dates.min() + pd.Timedelta(days=min_train_days)
    month_starts = pd.date_range(first_test.to_period("M").to_timestamp(), dates.max(), freq=refit)
    boundaries = [*month_starts, dates.max() + pd.Timedelta(days=1)]
    out = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        train = df[(dates < start) & df[y_col].notna()]
        test = df[(dates >= start) & (dates < end)]
        if test.empty or len(train) < min_train_days * 0.8:
            continue
        model = GaussianEMOS(members).fit(train, y_col)
        test = test.dropna(subset=members)
        mu, sigma = model.predict(test)
        out.append(test.assign(mu=mu, sigma=sigma, train_end=start))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
