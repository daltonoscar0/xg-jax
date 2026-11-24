"""Metrics, against values worked out by hand.

Every expectation here is computed from the definition in the test itself, not copied
from a previous run of the code it is testing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xg.metrics import (
    bootstrap_by_match,
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    log_loss,
    reliability_table,
    roc_auc,
    summarise,
)

# --------------------------------------------------------------------------------------
# scoring rules
# --------------------------------------------------------------------------------------


def test_log_loss_on_a_four_shot_toy_example():
    y = [1, 0, 1, 0]
    p = [0.8, 0.1, 0.6, 0.4]
    expected = -(math.log(0.8) + math.log(0.9) + math.log(0.6) + math.log(0.6)) / 4
    assert log_loss(y, p) == pytest.approx(expected)


def test_log_loss_of_a_perfect_prediction_is_zero():
    assert log_loss([1, 0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-8)


def test_log_loss_of_the_base_rate_is_the_entropy():
    """Predicting the base rate everywhere gives exactly the Bernoulli entropy."""
    y = np.array([1] * 20 + [0] * 80, dtype="float64")
    rate = y.mean()
    expected = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
    assert log_loss(y, np.full(100, rate)) == pytest.approx(expected)


def test_brier_on_the_same_toy_example():
    y = [1, 0, 1, 0]
    p = [0.8, 0.1, 0.6, 0.4]
    expected = (0.2 ** 2 + 0.1 ** 2 + 0.4 ** 2 + 0.4 ** 2) / 4
    assert brier_score(y, p) == pytest.approx(expected)


def test_auc_is_one_when_every_goal_outranks_every_miss():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_auc_is_a_half_when_all_predictions_are_tied():
    assert roc_auc([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]) == pytest.approx(0.5)


def test_auc_counts_ties_as_half_a_win():
    # One goal at 0.5, two misses at 0.4 and 0.5: one clear win, one tie.
    assert roc_auc([1, 0, 0], [0.5, 0.4, 0.5]) == pytest.approx((1 + 0.5) / 2)


def test_auc_matches_the_pairwise_definition_on_random_data():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(size=200)
    wins = 0.0
    for a in p[y == 1]:
        for b in p[y == 0]:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    assert roc_auc(y, p) == pytest.approx(wins / ((y == 1).sum() * (y == 0).sum()))


def test_auc_is_undefined_with_one_class():
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


def test_reliability_bins_are_equal_count_when_they_can_be():
    p = np.linspace(0.01, 0.99, 1000)
    y = (np.arange(1000) % 2).astype("float64")
    rows = reliability_table(y, p, n_bins=10, min_per_bin=50)
    assert len(rows) == 10
    assert {r["n"] for r in rows} == {100}


def test_short_bins_are_merged_rather_than_reported():
    """30 shots over 10 bins would be 3 per bin; the merge should give one usable bin."""
    rng = np.random.default_rng(1)
    p = rng.uniform(size=30)
    y = rng.integers(0, 2, size=30).astype("float64")
    rows = reliability_table(y, p, n_bins=10, min_per_bin=50)
    assert len(rows) == 1
    assert rows[0]["n"] == 30


def test_every_shot_lands_in_exactly_one_bin():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=437)
    y = rng.integers(0, 2, size=437).astype("float64")
    rows = reliability_table(y, p, n_bins=7, min_per_bin=20)
    assert sum(r["n"] for r in rows) == 437
    assert sum(r["goals"] for r in rows) == int(y.sum())


def test_calibration_error_is_zero_for_a_perfectly_calibrated_prediction():
    # 1000 shots at exactly 0.2, of which exactly 200 are goals.
    y = np.array([1.0] * 200 + [0.0] * 800)
    p = np.full(1000, 0.2)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-12)


def test_calibration_error_on_a_hand_computed_two_bin_example():
    # Bin one: 100 shots at 0.1, 20 goals -> gap 0.10. Bin two: 100 at 0.5, 40 -> 0.10.
    p = np.concatenate([np.full(100, 0.1), np.full(100, 0.5)])
    y = np.concatenate([np.array([1.0] * 20 + [0.0] * 80),
                        np.array([1.0] * 40 + [0.0] * 60)])
    assert expected_calibration_error(y, p, n_bins=2, min_per_bin=50) == \
        pytest.approx(0.10)


def test_calibration_slope_is_one_and_intercept_zero_for_honest_predictions():
    """Draw outcomes from the predictions themselves: the fit must recover 1 and 0."""
    rng = np.random.default_rng(5)
    p = rng.beta(1.2, 8.0, size=40000)
    y = (rng.uniform(size=p.size) < p).astype("float64")
    intercept, slope = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.06)
    assert intercept == pytest.approx(0.0, abs=0.06)


def test_calibration_slope_falls_below_one_when_predictions_are_over_spread():
    """Doubling the logits makes the model over-confident; the slope must notice."""
    rng = np.random.default_rng(6)
    p = rng.beta(1.2, 8.0, size=40000)
    y = (rng.uniform(size=p.size) < p).astype("float64")
    stretched = 1.0 / (1.0 + np.exp(-2.0 * np.log(p / (1 - p))))
    _, slope = calibration_slope_intercept(y, stretched)
    assert slope == pytest.approx(0.5, abs=0.06)


def test_calibration_slope_matches_a_closed_form_on_separable_data():
    # logit p = z, outcome exactly sigmoid(1.5 z): slope must come back 1.5.
    rng = np.random.default_rng(7)
    z = rng.normal(size=60000)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=z.size) < 1.0 / (1.0 + np.exp(-1.5 * z))).astype("float64")
    _, slope = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.5, abs=0.06)


def test_calibration_fit_does_not_overflow_on_extreme_predictions():
    """A near-separable prediction drives the Newton step into the tail of exp."""
    y = np.array([0.0] * 500 + [1.0] * 500)
    p = np.concatenate([np.full(500, 1e-9), np.full(500, 1.0 - 1e-9)])
    with np.errstate(over="raise"):
        intercept, slope = calibration_slope_intercept(y, p)
    assert np.isfinite(intercept) and np.isfinite(slope)


def test_summarise_reports_every_headline_metric():
    rng = np.random.default_rng(8)
    p = rng.beta(1.2, 8.0, size=3000)
    y = (rng.uniform(size=p.size) < p).astype("float64")
    out = summarise(y, p)
    assert set(out) == {"n_shots", "goals", "mean_pred", "log_loss", "brier", "auc",
                        "ece", "cal_intercept", "cal_slope"}
    assert out["n_shots"] == 3000
    assert out["goals"] == int(y.sum())
    assert out["mean_pred"] == pytest.approx(p.mean())


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(9)
    match_ids = np.repeat(np.arange(120), 12)
    p = rng.beta(1.2, 8.0, size=match_ids.size)
    y = (rng.uniform(size=p.size) < p).astype("float64")
    out = bootstrap_by_match(y, p, match_ids, n_boot=200, seed=1)
    for key in ("log_loss", "brier", "auc", "ece"):
        assert out[key]["lo"] <= out[key]["value"] <= out[key]["hi"]
    assert out["_n_boot"] > 150


def test_bootstrap_resamples_matches_not_shots():
    """Shot-level resampling gives a visibly narrower interval; this must not do that.

    With 40 matches of 25 highly correlated shots each, treating shots as independent
    understates the spread by roughly the square root of the cluster size.
    """
    rng = np.random.default_rng(10)
    n_matches, per_match = 40, 25
    match_ids = np.repeat(np.arange(n_matches), per_match)
    # Strong within-match correlation: each match has its own goal rate.
    rates = rng.uniform(0.02, 0.35, size=n_matches)
    p = np.repeat(rates, per_match)
    y = (rng.uniform(size=p.size) < p).astype("float64")

    by_match = bootstrap_by_match(y, p, match_ids, n_boot=400, seed=2)
    by_shot = bootstrap_by_match(y, p, np.arange(p.size), n_boot=400, seed=2)
    width = by_match["log_loss"]["hi"] - by_match["log_loss"]["lo"]
    narrow = by_shot["log_loss"]["hi"] - by_shot["log_loss"]["lo"]
    assert width > narrow * 1.5


def test_bootstrap_is_reproducible_from_its_seed():
    rng = np.random.default_rng(11)
    match_ids = np.repeat(np.arange(50), 10)
    p = rng.uniform(0.01, 0.6, size=match_ids.size)
    y = (rng.uniform(size=p.size) < p).astype("float64")
    a = bootstrap_by_match(y, p, match_ids, n_boot=100, seed=42)
    b = bootstrap_by_match(y, p, match_ids, n_boot=100, seed=42)
    assert a["log_loss"] == b["log_loss"]
