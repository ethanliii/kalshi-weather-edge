import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from weather_edge.kalshi.markets import Bracket
from weather_edge.model.brackets import probs_members, probs_normal
from weather_edge.model.emos import GaussianEMOS, crps_normal, walk_forward


def make_brackets(first_lo: int, n_between: int, width: int = 2) -> list[Bracket]:
    bs = [Bracket("lo", -math.inf, first_lo - 1)]
    for i in range(n_between):
        lo = first_lo + i * width
        bs.append(Bracket(f"b{i}", lo, lo + width - 1))
    bs.append(Bracket("hi", first_lo + n_between * width, math.inf))
    return bs


# ----------------------------------------------------------------- sum to one
@pytest.mark.parametrize("seed", range(25))
def test_normal_bracket_probs_sum_to_one(seed):
    rng = np.random.default_rng(seed)
    bs = make_brackets(int(rng.integers(20, 100)), int(rng.integers(1, 8)), int(rng.integers(1, 4)))
    p = probs_normal(rng.normal(60, 25), rng.uniform(0.3, 10), bs)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p >= 0).all()


@pytest.mark.parametrize("seed", range(25))
def test_member_bracket_probs_sum_to_one(seed):
    rng = np.random.default_rng(seed)
    bs = make_brackets(70, 4)
    p = probs_members(rng.normal(75, 6, size=int(rng.integers(1, 60))), bs)
    assert p.sum() == pytest.approx(1.0)


def test_far_tail_mass_goes_to_tail_bracket():
    bs = make_brackets(70, 4)
    p = probs_normal(150, 2, bs)
    assert p[-1] == pytest.approx(1.0) and p[:-1].sum() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------- correct discretisation
def test_bracket_uses_half_degree_edges():
    bs = make_brackets(70, 4)  # ≤69, 70-71, 72-73, 74-75, 76-77, ≥78
    mu, sd = 72.3, 1.7
    p = probs_normal(mu, sd, bs)
    expected_72_73 = norm.cdf(73.5, mu, sd) - norm.cdf(71.5, mu, sd)
    assert p[2] == pytest.approx(expected_72_73)
    assert p[0] == pytest.approx(norm.cdf(69.5, mu, sd))


def test_members_round_half_up_like_cli():
    bs = make_brackets(70, 4)
    assert probs_members([69.5], bs)[1] == 1.0  # reported 70 -> bracket 70-71
    assert probs_members([69.49], bs)[0] == 1.0  # reported 69 -> ≤69


def test_invalid_inputs():
    bs = make_brackets(70, 2)
    with pytest.raises(ValueError):
        probs_normal(70, 0, bs)
    with pytest.raises(ValueError):
        probs_members([np.nan], bs)
    with pytest.raises(ValueError):
        probs_normal(70, 1, bs[1:])  # not a partition


# ------------------------------------------------------------------- CRPS
@pytest.mark.parametrize("mu,sigma,y", [(0, 1, 0), (2, 3, -1), (70, 2.5, 74)])
def test_crps_matches_numerical_integral(mu, sigma, y):
    integrand = lambda x: (norm.cdf(x, mu, sigma) - (x >= y)) ** 2  # noqa: E731
    numeric = integrate.quad(integrand, mu - 12 * sigma, y)[0] + integrate.quad(integrand, y, mu + 12 * sigma)[0]
    assert crps_normal(np.array(mu), np.array(sigma), np.array(y)) == pytest.approx(numeric, rel=1e-6)


def test_crps_tends_to_absolute_error():
    assert crps_normal(np.array(5.0), np.array(1e-6), np.array(8.0)) == pytest.approx(3.0, abs=1e-4)


# ------------------------------------------------------------------- EMOS
def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    truth_signal = 60 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + rng.normal(0, 6, n)
    # Two biased, noisy "models"; the observation has heteroscedastic noise tied to model disagreement.
    disagreement = rng.uniform(0.2, 3, n)
    m1 = truth_signal - 2 + rng.normal(0, 0.5, n) + disagreement
    m2 = truth_signal - 2 + rng.normal(0, 0.5, n) - disagreement
    obs = truth_signal + rng.normal(0, 1 + 0.8 * disagreement)
    return pd.DataFrame({"date": dates, "m1": m1, "m2": m2, "obs": obs})


def test_emos_removes_bias_and_is_calibrated():
    df = synthetic()
    train, test = df.iloc[:600], df.iloc[600:]
    model = GaussianEMOS(["m1", "m2"]).fit(train)
    mu, sigma = model.predict(test)
    resid = test["obs"].to_numpy() - mu
    assert abs(resid.mean()) < 0.3  # the -2 bias is learned
    pit = norm.cdf(test["obs"].to_numpy(), mu, sigma)
    coverage_80 = np.mean((pit > 0.1) & (pit < 0.9))
    assert 0.72 < coverage_80 < 0.88
    # spread should track model disagreement
    assert np.corrcoef(sigma, test["m1"] - test["m2"])[0, 1] > 0.5


def test_emos_beats_raw_ensemble_crps():
    df = synthetic(seed=1)
    train, test = df.iloc[:600], df.iloc[600:]
    mu, sigma = GaussianEMOS(["m1", "m2"]).fit(train).predict(test)
    raw_mu = test[["m1", "m2"]].mean(axis=1).to_numpy()
    raw_sd = test[["m1", "m2"]].std(axis=1).to_numpy() + 1e-3
    y = test["obs"].to_numpy()
    assert crps_normal(mu, sigma, y).mean() < crps_normal(raw_mu, raw_sd, y).mean()


def test_emos_requires_data_and_fit():
    with pytest.raises(ValueError):
        GaussianEMOS(["m1"]).fit(synthetic(n=10))
    with pytest.raises(RuntimeError):
        GaussianEMOS(["m1"]).predict(synthetic(n=10))


def test_walk_forward_is_out_of_sample():
    df = synthetic(n=500, seed=2)
    preds = walk_forward(df, ["m1", "m2"], min_train_days=200)
    assert not preds.empty
    assert (pd.to_datetime(preds["date"]) >= preds["train_end"]).all()
    # Corrupting the future must not change past predictions.
    corrupted = df.copy()
    corrupted.loc[corrupted["date"] >= "2025-03-01", "obs"] += 50
    preds2 = walk_forward(corrupted, ["m1", "m2"], min_train_days=200)
    before = preds["date"] < "2025-03-01"
    np.testing.assert_allclose(preds.loc[before, "mu"], preds2.loc[before, "mu"])
