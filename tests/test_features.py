"""Features: the leakage guard, freeze frames, and train-only standardisation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from xg.features import (
    CONTINUOUS,
    FeatureStats,
    LeakageError,
    assert_no_leakage,
    baseline_matrix,
    build_frame,
    column_names,
    design_matrix,
    freeze_frame_features,
)

# --------------------------------------------------------------------------------------
# the leakage guard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("column", [
    "goal", "outcome", "shot_outcome", "shot_statsbomb_xg", "statsbomb_xg",
    "end_x", "shot_end_location", "shot_deflected", "shot_saved_to_post",
    "next_event", "related_events", "post_shot_xg", "is_goal",
])
def test_leakage_guard_refuses_outcome_columns(column):
    with pytest.raises(LeakageError):
        assert_no_leakage(["distance", "angle", column])


def test_leakage_guard_refuses_an_unseen_name_that_smells_of_the_outcome():
    # Not in the explicit list: caught by the substring scan.
    with pytest.raises(LeakageError):
        assert_no_leakage(["distance", "my_shot_outcome_flag"])
    with pytest.raises(LeakageError):
        assert_no_leakage(["distance", "deflection_angle"])


def test_leakage_guard_allows_the_innocent_uses_of_the_word_goal():
    assert_no_leakage(["goal_distance", "goal_angle", "gk_dist_goal_line",
                       "play_pattern_from_goal_kick", "shot_type_open_goal"])


def test_the_real_design_matrix_passes_its_own_guard():
    """The guard is only worth having if it is actually run on the shipped columns."""
    assert_no_leakage(column_names())


def test_design_matrix_contains_no_outcome_column(fixture_shots):
    raw = build_frame(fixture_shots)
    stats = FeatureStats.fit(raw)
    names = set(stats.columns)
    assert "goal" not in names
    assert not any("statsbomb" in n for n in names)
    assert design_matrix(raw, stats).shape == (len(fixture_shots), len(names))


# --------------------------------------------------------------------------------------
# freeze frames
# --------------------------------------------------------------------------------------


def test_freeze_frame_absent_gives_all_nan():
    out = freeze_frame_features("[]", 90.0, 34.0)
    assert all(np.isnan(v) for v in out.values())
    assert all(np.isnan(v) for v in freeze_frame_features(None, 90.0, 34.0).values())


def test_freeze_frame_counts_only_opponents_in_the_triangle():
    shot_x, shot_y = 90.0, 34.0
    frame = [
        # In the triangle, an opponent: counts.
        {"teammate": False, "position": "Centre Back", "location": [97.0, 34.0]},
        # In the triangle but a teammate: does not count.
        {"teammate": True, "position": "Centre Forward", "location": [98.0, 34.5]},
        # An opponent, but wide of the triangle.
        {"teammate": False, "position": "Left Back", "location": [97.0, 12.0]},
        # An opponent behind the shooter.
        {"teammate": False, "position": "Right Back", "location": [84.0, 34.0]},
    ]
    out = freeze_frame_features(json.dumps(frame), shot_x, shot_y)
    assert out["n_opponents_in_triangle"] == 1.0


def test_keeper_features_are_measured_from_the_goal_line_and_the_shot_line():
    # Keeper 3 m off his line and dead on the shot line.
    frame = [{"teammate": False, "position": "Goalkeeper", "location": [102.0, 34.0]}]
    out = freeze_frame_features(json.dumps(frame), 90.0, 34.0)
    assert out["gk_dist_goal_line"] == pytest.approx(3.0)
    assert out["gk_dist_shot_line"] == pytest.approx(0.0)
    # No outfield opponent in the frame, so there is no nearest defender.
    assert np.isnan(out["nearest_defender_dist"])


def test_nearest_defender_excludes_the_keeper():
    frame = [
        {"teammate": False, "position": "Goalkeeper", "location": [91.0, 34.0]},
        {"teammate": False, "position": "Centre Back", "location": [95.0, 34.0]},
    ]
    out = freeze_frame_features(json.dumps(frame), 90.0, 34.0)
    # The keeper is 1 m away, the nearest outfielder 5 m.
    assert out["nearest_defender_dist"] == pytest.approx(5.0)


def test_malformed_freeze_frame_does_not_raise():
    assert np.isnan(freeze_frame_features("{not json", 90.0, 34.0)["gk_dist_goal_line"])


# --------------------------------------------------------------------------------------
# standardisation
# --------------------------------------------------------------------------------------


def test_standardisation_uses_training_statistics_only(fixture_shots):
    """Fit on one half, transform the other: the second half must not be re-centred."""
    raw = build_frame(fixture_shots)
    first, second = raw.iloc[:200], raw.iloc[200:]
    stats = FeatureStats.fit(first)

    train_matrix = stats.transform(first)
    held_matrix = stats.transform(second)
    distance_col = list(stats.columns).index("distance")

    assert train_matrix[:, distance_col].mean() == pytest.approx(0.0, abs=1e-5)
    # The held-out half is standardised with the training constants, so its own mean is
    # whatever it is -- not zero. If it came out at zero, the stats were refitted.
    assert abs(held_matrix[:, distance_col].mean()) > 1e-4


def test_missing_continuous_values_are_imputed_with_the_training_median(fixture_shots):
    raw = build_frame(fixture_shots)
    stats = FeatureStats.fit(raw)
    no_frame = raw.loc[~raw["has_freeze_frame"].astype(bool)]
    assert len(no_frame) > 0

    matrix = stats.transform(no_frame)
    col = list(stats.columns).index("nearest_defender_dist")
    expected = (stats.median["nearest_defender_dist"] - stats.mean["nearest_defender_dist"]) \
        / stats.std["nearest_defender_dist"]
    assert matrix[:, col] == pytest.approx(expected, abs=1e-5)


def test_the_matrix_is_finite_everywhere(fixture_shots):
    raw = build_frame(fixture_shots)
    stats = FeatureStats.fit(raw)
    assert np.isfinite(stats.transform(raw)).all()
    assert np.isfinite(baseline_matrix(raw, stats)).all()


def test_baseline_matrix_is_distance_and_angle_only(fixture_shots):
    raw = build_frame(fixture_shots)
    stats = FeatureStats.fit(raw)
    assert baseline_matrix(raw, stats).shape == (len(raw), 2)


def test_column_order_is_stable(fixture_shots):
    raw = build_frame(fixture_shots)
    a = FeatureStats.fit(raw).columns
    b = FeatureStats.fit(raw.iloc[::-1]).columns
    assert a == b
    assert len(a) == len(set(a))


def test_every_continuous_feature_is_present_in_the_frame(fixture_shots):
    raw = build_frame(fixture_shots)
    for name in CONTINUOUS:
        assert name in raw.columns


def test_set_piece_flag_follows_the_shot_type(fixture_shots):
    raw = build_frame(fixture_shots)
    is_set_piece = raw["is_set_piece"].to_numpy().astype(bool)
    from_free_kick = fixture_shots["shot_type"].eq("free_kick").to_numpy()
    assert is_set_piece[from_free_kick].all()
    assert not is_set_piece[fixture_shots["shot_type"].eq("open_play").to_numpy()].any()


def test_unknown_categorical_levels_collapse_to_other(fixture_shots):
    shots = fixture_shots.copy()
    shots.loc[shots.index[0], "body_part"] = "elbow"
    raw = build_frame(shots)
    assert raw["body_part"].iloc[0] == "other"
