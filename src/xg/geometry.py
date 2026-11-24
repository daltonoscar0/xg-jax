"""Shot geometry on the canonical 105 x 68 m pitch.

Everything downstream works in metres on sportio's canonical pitch: origin bottom-left,
the attacking team shooting towards +x, so the goal being shot at is always the one at
``x = 105`` with its centre at ``y = 34``.

StatsBomb publishes coordinates on a nominal 120 x 80 grid that is *not* 105 x 68 in
proportion, so a plain linear rescale shrinks the goal to 6.80 m and the penalty spot to
10.50 m. We use sportio's landmark spec instead, which pins the goal line, goal area,
penalty area, penalty spot and halfway line to their real positions. Box-relative
distance is the single most important xG feature, so the 5-7% matters.
"""

from __future__ import annotations

import numpy as np
from sportio import PITCH_LENGTH, PITCH_WIDTH, STATSBOMB_LANDMARK

__all__ = [
    "GOAL_CENTRE",
    "GOAL_WIDTH",
    "LEFT_POST",
    "RIGHT_POST",
    "goal_angle",
    "goal_distance",
    "in_shot_triangle",
    "point_segment_distance",
    "statsbomb_to_canonical",
]

#: Laws of the Game: the goal is 7.32 m between the posts.
GOAL_WIDTH = 7.32
GOAL_CENTRE = (PITCH_LENGTH, PITCH_WIDTH / 2.0)
#: Posts as seen by a player shooting towards +x; "left"/"right" are just the two ends.
LEFT_POST = (PITCH_LENGTH, PITCH_WIDTH / 2.0 - GOAL_WIDTH / 2.0)
RIGHT_POST = (PITCH_LENGTH, PITCH_WIDTH / 2.0 + GOAL_WIDTH / 2.0)


def statsbomb_to_canonical(x, y):
    """Map StatsBomb 120 x 80 coordinates onto the canonical 105 x 68 m pitch."""
    return STATSBOMB_LANDMARK.to_canonical(np.asarray(x, dtype="float64"),
                                           np.asarray(y, dtype="float64"))


def goal_distance(x, y):
    """Straight-line distance in metres from the shot to the centre of the goal."""
    dx = PITCH_LENGTH - np.asarray(x, dtype="float64")
    dy = np.asarray(y, dtype="float64") - PITCH_WIDTH / 2.0
    return np.hypot(dx, dy)


def goal_angle(x, y):
    """Angle in radians subtended at the shot location by the two goalposts.

    Derived from the cross and dot products of the vectors to each post, which collapses
    to ``atan2(w * dx, dx^2 + dy^2 - (w/2)^2)``. ``atan2`` keeps the obtuse case right:
    behind the goal line between the posts the angle is more than 90 degrees, and standing
    exactly on the goal line at the centre it is pi.
    """
    dx = PITCH_LENGTH - np.asarray(x, dtype="float64")
    dy = np.asarray(y, dtype="float64") - PITCH_WIDTH / 2.0
    half = GOAL_WIDTH / 2.0
    return np.arctan2(GOAL_WIDTH * dx, dx * dx + dy * dy - half * half)


def in_shot_triangle(shot_x, shot_y, px, py):
    """Which of the points ``(px, py)`` fall inside the shot/left-post/right-post triangle.

    Vectorised over the points for one shot. Uses the sign of the three edge cross
    products; a point on an edge counts as inside.
    """
    px = np.asarray(px, dtype="float64")
    py = np.asarray(py, dtype="float64")
    verts = [(float(shot_x), float(shot_y)), LEFT_POST, RIGHT_POST]
    signs = []
    for (ax, ay), (bx, by) in zip(verts, verts[1:] + verts[:1], strict=True):
        signs.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    signs = np.stack(signs)
    tol = 1e-9
    return np.all(signs >= -tol, axis=0) | np.all(signs <= tol, axis=0)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Perpendicular distance from points ``(px, py)`` to the segment ``a -> b``.

    Used for "how far off the shot line is the keeper", where the segment is the shot
    location to the centre of the goal.
    """
    px = np.asarray(px, dtype="float64")
    py = np.asarray(py, dtype="float64")
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom == 0.0:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))
