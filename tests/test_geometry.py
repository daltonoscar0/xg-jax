"""Hand-computed geometry.

Every expected value here is worked out on paper from the laws of the game, not read off
a previous run of the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xg.geometry import (
    GOAL_WIDTH,
    goal_angle,
    goal_distance,
    in_shot_triangle,
    point_segment_distance,
    statsbomb_to_canonical,
)

GOAL_X, GOAL_Y = 105.0, 34.0


def test_distance_from_the_centre_of_the_goal_line_is_zero():
    assert goal_distance(GOAL_X, GOAL_Y) == pytest.approx(0.0)


def test_angle_from_the_centre_of_the_goal_line_is_half_a_turn():
    # Standing on the goal line between the posts, one post is to your left and the
    # other to your right: the angle they subtend is pi.
    assert goal_angle(GOAL_X, GOAL_Y) == pytest.approx(math.pi)


def test_penalty_spot_distance_is_eleven_metres():
    # The penalty mark is 11 m from the goal line, on the centre line of the goal.
    assert goal_distance(GOAL_X - 11.0, GOAL_Y) == pytest.approx(11.0)


def test_penalty_spot_angle():
    # Half the goal is 3.66 m across at 11 m out, so each half subtends atan(3.66/11)
    # and the whole goal twice that.
    expected = 2.0 * math.atan((GOAL_WIDTH / 2.0) / 11.0)
    assert goal_angle(GOAL_X - 11.0, GOAL_Y) == pytest.approx(expected)
    assert math.degrees(expected) == pytest.approx(36.81, abs=0.01)


def test_corner_flag_distance_is_half_the_pitch_width():
    # The corner flag sits on the goal line at the touchline: 34 m from the goal centre.
    assert goal_distance(GOAL_X, 0.0) == pytest.approx(34.0)
    assert goal_distance(GOAL_X, 68.0) == pytest.approx(34.0)


def test_corner_flag_angle_is_zero():
    # From a point on the goal line outside the posts, both posts are in the same
    # direction, so they subtend nothing.
    assert goal_angle(GOAL_X, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_angle_is_symmetric_about_the_goal_centre_line():
    for dx, dy in ((16.0, 8.0), (30.0, 20.0), (5.0, 2.0)):
        left = goal_angle(GOAL_X - dx, GOAL_Y - dy)
        right = goal_angle(GOAL_X - dx, GOAL_Y + dy)
        assert left == pytest.approx(right)


def test_angle_falls_away_with_distance_straight_on():
    straight = [goal_angle(GOAL_X - d, GOAL_Y) for d in (6.0, 11.0, 20.0, 35.0)]
    assert straight == sorted(straight, reverse=True)


def test_angle_matches_the_cosine_rule_at_an_arbitrary_point():
    # Independent derivation: the angle between the two vectors to the posts.
    x, y = 92.0, 26.0
    v1 = np.array([GOAL_X - x, (GOAL_Y - GOAL_WIDTH / 2) - y])
    v2 = np.array([GOAL_X - x, (GOAL_Y + GOAL_WIDTH / 2) - y])
    cosine = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert goal_angle(x, y) == pytest.approx(math.acos(cosine))


def test_statsbomb_landmarks_land_on_the_real_pitch_furniture():
    # StatsBomb's own 120x80 grid: (120, 40) is the goal centre, (108, 40) the penalty
    # spot, (120, 80) a corner flag, (120, 36) and (120, 44) the two posts. StatsBomb's
    # y axis points down, so y=36 comes out as the *far* post, 34 + 3.66.
    x, y = statsbomb_to_canonical([120.0, 108.0, 120.0, 120.0, 120.0],
                                  [40.0, 40.0, 80.0, 36.0, 44.0])
    assert x[0] == pytest.approx(105.0)
    assert y[0] == pytest.approx(34.0)
    assert goal_distance(x[1], y[1]) == pytest.approx(11.0, abs=1e-6)
    assert goal_distance(x[2], y[2]) == pytest.approx(34.0, abs=1e-6)
    assert y[3] == pytest.approx(34.0 + GOAL_WIDTH / 2, abs=1e-6)
    assert y[4] == pytest.approx(34.0 - GOAL_WIDTH / 2, abs=1e-6)
    # Both posts, so a shot on the goal line between them subtends the whole goal.
    assert abs(y[3] - y[4]) == pytest.approx(GOAL_WIDTH, abs=1e-6)


def test_vectorised_over_arrays():
    xs = np.array([105.0, 94.0, 105.0])
    ys = np.array([34.0, 34.0, 0.0])
    assert goal_distance(xs, ys) == pytest.approx([0.0, 11.0, 34.0])
    assert goal_angle(xs, ys) == pytest.approx([math.pi, 2 * math.atan(3.66 / 11.0), 0.0])


def test_shot_triangle_contains_a_defender_on_the_line_and_excludes_one_behind_the_shooter():
    shot_x, shot_y = 90.0, 34.0
    inside = in_shot_triangle(shot_x, shot_y, [97.0], [34.0])
    behind = in_shot_triangle(shot_x, shot_y, [85.0], [34.0])
    wide = in_shot_triangle(shot_x, shot_y, [97.0], [10.0])
    assert bool(inside[0]) is True
    assert bool(behind[0]) is False
    assert bool(wide[0]) is False


def test_shot_triangle_vertices_count_as_inside():
    shot_x, shot_y = 90.0, 34.0
    on_post = in_shot_triangle(shot_x, shot_y, [105.0], [34.0 - GOAL_WIDTH / 2])
    assert bool(on_post[0]) is True


def test_keeper_on_the_shot_line_has_zero_offset():
    # Shot at (90, 34), goal centre (105, 34): a keeper at (99, 34) is dead on the line.
    assert point_segment_distance(99.0, 34.0, 90.0, 34.0, 105.0, 34.0) == pytest.approx(0.0)
    # Three metres off it, perpendicular.
    assert point_segment_distance(99.0, 37.0, 90.0, 34.0, 105.0, 34.0) == pytest.approx(3.0)
