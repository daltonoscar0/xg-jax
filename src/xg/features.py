"""Shot features, every one of them knowable at the instant the ball is struck.

Two rules are enforced here rather than left to discipline:

* :func:`assert_no_leakage` refuses any column that describes the outcome of the shot or
  anything that happened after it (``goal``, ``outcome``, ``shot_deflected``, the end
  location of the ball, and StatsBomb's own xG).
* :class:`FeatureStats` is fitted on the training split only and then applied unchanged
  to validation, test and the held-out competition. Standardisation constants and the
  medians used to impute missing freeze-frame values never see held-out data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geometry import (
    GOAL_CENTRE,
    goal_angle,
    goal_distance,
    in_shot_triangle,
    point_segment_distance,
)

__all__ = [
    "CATEGORICALS",
    "CONTINUOUS",
    "FLAGS",
    "LEAKY_COLUMNS",
    "FeatureStats",
    "LeakageError",
    "assert_no_leakage",
    "baseline_matrix",
    "build_frame",
    "column_names",
    "design_matrix",
    "freeze_frame_features",
]


class LeakageError(ValueError):
    """Raised when a feature column would tell the model what happened to the shot."""


#: Columns that are the answer, part of the answer, or a competitor's answer.
LEAKY_COLUMNS = frozenset({
    "goal", "outcome", "shot_outcome", "shot_statsbomb_xg", "statsbomb_xg",
    "end_x", "end_y", "end_z", "shot_end_location", "end_location",
    "shot_deflected", "shot_saved_to_post", "shot_saved_off_target",
    "goals", "is_goal", "xg", "next_event", "related_events",
})

#: Substrings that make a column suspect whatever it is called.
_LEAKY_PATTERNS = ("outcome", "end_location", "statsbomb_xg", "_xg", "goal_scored",
                   "deflect", "saved_", "post_shot", "next_")

#: Columns exempt from the substring scan because the word appears innocently.
_LEAKY_EXEMPT = frozenset({"goal_distance", "goal_angle", "distance_to_goal",
                           "gk_dist_goal_line", "pass_goal_assist", "play_pattern_from_goal_kick",
                           "shot_type_open_goal", "shot_open_goal"})

#: Continuous features, standardised with training-set mean and standard deviation.
CONTINUOUS = (
    "distance", "angle", "x", "y", "lateral_offset",
    "n_opponents_in_triangle", "gk_dist_goal_line", "gk_dist_shot_line",
    "nearest_defender_dist", "pass_length",
)

#: Binary flags, left at 0/1.
FLAGS = (
    "shot_first_time", "shot_one_on_one", "shot_open_goal", "shot_aerial_won",
    "shot_follows_dribble", "shot_redirect", "under_pressure", "is_set_piece",
    "has_freeze_frame", "has_key_pass",
)

#: Categoricals, one-hot encoded with the level list fixed at fit time.
CATEGORICALS = {
    "body_part": ("right_foot", "left_foot", "head", "other"),
    "shot_type": ("open_play", "free_kick", "corner", "kick_off", "other"),
    "play_pattern": ("regular_play", "from_corner", "from_free_kick", "from_throw_in",
                     "from_counter", "from_goal_kick", "from_keeper", "from_kick_off",
                     "other"),
    "technique": ("normal", "half_volley", "volley", "lob", "overhead_kick", "backheel",
                  "diving_header", "other"),
    "assist_type": ("none", "through_ball", "cross", "cutback", "other"),
}

#: Set-piece play patterns and shot types, collapsed into one flag.
_SET_PIECE_TYPES = frozenset({"free_kick", "corner", "kick_off"})


def assert_no_leakage(columns) -> None:
    """Refuse a feature set that contains outcome or post-shot information."""
    bad = []
    for col in columns:
        name = str(col)
        if name in _LEAKY_EXEMPT:
            continue
        if name in LEAKY_COLUMNS:
            bad.append(name)
            continue
        lowered = name.lower()
        if any(pattern in lowered for pattern in _LEAKY_PATTERNS):
            bad.append(name)
    if bad:
        raise LeakageError(
            "these columns describe the shot's outcome or a later event and cannot be "
            f"features: {sorted(set(bad))}"
        )


# --------------------------------------------------------------------------------------
# freeze frames
# --------------------------------------------------------------------------------------


def freeze_frame_features(frame: str | list | None, shot_x: float, shot_y: float) -> dict:
    """Opponent geometry from StatsBomb's shot freeze frame.

    Returns NaN for every value when the shot has no freeze frame; the caller imputes
    with a training-set median and flags the row with ``has_freeze_frame``.
    """
    missing = {"n_opponents_in_triangle": np.nan, "gk_dist_goal_line": np.nan,
               "gk_dist_shot_line": np.nan, "nearest_defender_dist": np.nan}
    players = frame if isinstance(frame, list) else _loads(frame)
    if not players:
        return missing

    opponents = [p for p in players if not p.get("teammate")]
    if not opponents:
        return missing

    locs = np.array([p.get("location") or [np.nan, np.nan] for p in opponents], dtype="float64")
    is_keeper = np.array([(p.get("position") or "") == "Goalkeeper" for p in opponents])

    out = dict(missing)
    good = np.isfinite(locs).all(axis=1)
    if not good.any():
        return out

    px, py = locs[good, 0], locs[good, 1]
    out["n_opponents_in_triangle"] = float(in_shot_triangle(shot_x, shot_y, px, py).sum())

    # "Nearest defender" means nearest outfielder: the keeper gets his own two features.
    outfield = good & ~is_keeper
    if outfield.any():
        out["nearest_defender_dist"] = float(
            np.hypot(locs[outfield, 0] - shot_x, locs[outfield, 1] - shot_y).min()
        )

    keeper = good & is_keeper
    if keeper.any():
        kx, ky = locs[keeper, 0][0], locs[keeper, 1][0]
        out["gk_dist_goal_line"] = float(GOAL_CENTRE[0] - kx)
        out["gk_dist_shot_line"] = float(
            point_segment_distance(kx, ky, shot_x, shot_y, GOAL_CENTRE[0], GOAL_CENTRE[1])
        )
    return out


def _loads(blob: object) -> list:
    if not isinstance(blob, str) or not blob:
        return []
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


# --------------------------------------------------------------------------------------
# the raw feature frame
# --------------------------------------------------------------------------------------


def build_frame(shots: pd.DataFrame) -> pd.DataFrame:
    """Raw (unstandardised) features, one row per shot, in the shots' own order."""
    out = pd.DataFrame(index=shots.index)
    x = shots["x"].astype("float64").to_numpy()
    y = shots["y"].astype("float64").to_numpy()

    out["distance"] = goal_distance(x, y)
    out["angle"] = goal_angle(x, y)
    out["x"] = x
    out["y"] = y
    # The pitch is symmetric about the goal line, so the signed y cannot carry the effect
    # of being wide on its own in a linear model.
    out["lateral_offset"] = np.abs(y - GOAL_CENTRE[1])

    ff = [freeze_frame_features(blob, sx, sy)
          for blob, sx, sy in zip(shots["freeze_frame"], x, y, strict=True)]
    for key in ("n_opponents_in_triangle", "gk_dist_goal_line", "gk_dist_shot_line",
                "nearest_defender_dist"):
        out[key] = np.array([row[key] for row in ff], dtype="float64")

    out["pass_length"] = pd.to_numeric(shots["pass_length"], errors="coerce").to_numpy()

    for flag in FLAGS:
        if flag == "is_set_piece":
            out[flag] = shots["shot_type"].isin(_SET_PIECE_TYPES).astype("float64").to_numpy()
        else:
            out[flag] = shots[flag].fillna(False).astype(bool).astype("float64").to_numpy()

    out["body_part"] = _level(shots["body_part"], CATEGORICALS["body_part"])
    out["shot_type"] = _level(shots["shot_type"], CATEGORICALS["shot_type"])
    out["play_pattern"] = _level(shots["play_pattern"], CATEGORICALS["play_pattern"])
    out["technique"] = _level(shots["technique"], CATEGORICALS["technique"])
    out["assist_type"] = _assist_type(shots)
    return out


def _level(series: pd.Series, levels: tuple[str, ...]) -> np.ndarray:
    values = series.fillna("other").astype(str).to_numpy()
    allowed = set(levels)
    return np.array([v if v in allowed else "other" for v in values], dtype=object)


def _assist_type(shots: pd.DataFrame) -> np.ndarray:
    """Collapse the key pass into one of through ball / cross / cutback / other / none."""
    out = np.full(len(shots), "none", dtype=object)
    has = shots["has_key_pass"].fillna(False).astype(bool).to_numpy()
    out[has] = "other"
    for flag, label in (("pass_cross", "cross"), ("pass_cut_back", "cutback"),
                        ("pass_through_ball", "through_ball")):
        mask = has & shots[flag].fillna(False).astype(bool).to_numpy()
        out[mask] = label
    return out


# --------------------------------------------------------------------------------------
# standardisation, fitted on train only
# --------------------------------------------------------------------------------------


@dataclass
class FeatureStats:
    """Means, standard deviations and imputation medians from the training split."""

    columns: tuple[str, ...] = ()
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, raw: pd.DataFrame) -> FeatureStats:
        median = {c: float(np.nanmedian(raw[c].to_numpy(dtype="float64")))
                  for c in CONTINUOUS}
        median = {c: (0.0 if not np.isfinite(v) else v) for c, v in median.items()}
        filled = {c: np.where(np.isfinite(raw[c].to_numpy(dtype="float64")),
                              raw[c].to_numpy(dtype="float64"), median[c])
                  for c in CONTINUOUS}
        mean = {c: float(v.mean()) for c, v in filled.items()}
        std = {c: float(v.std()) or 1.0 for c, v in filled.items()}
        stats = cls(mean=mean, std=std, median=median)
        stats.columns = tuple(column_names())
        return stats

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        """Standardised, one-hot-expanded design matrix in a fixed column order."""
        blocks = []
        for col in CONTINUOUS:
            v = raw[col].to_numpy(dtype="float64")
            v = np.where(np.isfinite(v), v, self.median[col])
            blocks.append((v - self.mean[col]) / self.std[col])
        for col in FLAGS:
            blocks.append(raw[col].to_numpy(dtype="float64"))
        for col, levels in CATEGORICALS.items():
            values = raw[col].to_numpy()
            # Drop the first level as the reference category so the design is full rank.
            for level in levels[1:]:
                blocks.append((values == level).astype("float64"))
        matrix = np.stack(blocks, axis=1).astype("float32")
        assert_no_leakage(self.columns)
        return matrix


def column_names() -> list[str]:
    """The design matrix's columns, in the order :meth:`FeatureStats.transform` emits."""
    names = list(CONTINUOUS) + list(FLAGS)
    for col, levels in CATEGORICALS.items():
        names += [f"{col}_{level}" for level in levels[1:]]
    return names


def design_matrix(raw: pd.DataFrame, stats: FeatureStats) -> np.ndarray:
    """All features."""
    return stats.transform(raw)


def baseline_matrix(raw: pd.DataFrame, stats: FeatureStats) -> np.ndarray:
    """Distance and angle only, standardised with the same training constants."""
    blocks = []
    for col in ("distance", "angle"):
        v = raw[col].to_numpy(dtype="float64")
        v = np.where(np.isfinite(v), v, stats.median[col])
        blocks.append((v - stats.mean[col]) / stats.std[col])
    return np.stack(blocks, axis=1).astype("float32")


BASELINE_COLUMNS = ("distance", "angle")
