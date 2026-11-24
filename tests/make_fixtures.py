#!/usr/bin/env python3
"""Regenerate ``tests/fixtures/shots.json``.

The fixture is synthetic: locations, body parts and freeze frames are drawn from a
generator here, and goals are drawn from a plausible hand-written xG function. It is not
StatsBomb data and must not be replaced with any, because the StatsBomb Public Data User
Agreement does not permit redistributing theirs.

    python tests/make_fixtures.py

Deterministic given the seed, so the committed file only changes when this script does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SEED = 4242
N_MATCHES = 24
OUT = Path(__file__).parent / "fixtures" / "shots.json"

COMPETITIONS = [
    (1, 11, "Invented League", "2024/2025", "male"),
    (1, 12, "Invented League", "2025/2026", "male"),
    (2, 21, "Invented Cup", "2025/2026", "female"),
]

BODY_PARTS = ["right_foot", "left_foot", "head", "other"]
BODY_WEIGHTS = [0.58, 0.26, 0.14, 0.02]
SHOT_TYPES = ["open_play", "free_kick", "corner"]
SHOT_WEIGHTS = [0.88, 0.07, 0.05]
PATTERNS = ["regular_play", "from_corner", "from_free_kick", "from_counter",
            "from_throw_in"]
TECHNIQUES = ["normal", "half_volley", "volley", "lob"]
PLAYERS = [f"Player {chr(65 + i // 3)}{i % 3 + 1}" for i in range(30)]
TEAMS = ["Invented FC", "Fictional United", "Placeholder City", "Example Rovers"]


def true_xg(distance: float, angle: float, body_part: str, set_piece: bool) -> float:
    """A hand-written, deliberately simple goal probability.

    Shaped like a real xG curve -- falling fast with distance, rising with the angle, a
    penalty for headers -- so the fixture exercises the same code paths as real shots
    without anyone mistaking it for real data.
    """
    logit = 0.9 + 1.9 * math.log(max(angle, 1e-3)) - 0.09 * distance
    if body_part == "head":
        logit -= 0.55
    if set_piece:
        logit -= 0.35
    return 1.0 / (1.0 + math.exp(-logit))


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []
    shot_id = 0

    for match_number in range(N_MATCHES):
        cid, sid, comp, season, gender = COMPETITIONS[match_number % len(COMPETITIONS)]
        match_id = 900_000 + match_number
        home, away = rng.choice(TEAMS, size=2, replace=False)

        for _ in range(int(rng.integers(14, 30))):
            shot_id += 1
            # Clustered near the top of the box, with a long tail outside it.
            x = float(np.clip(105.0 - rng.gamma(3.0, 5.0), 55.0, 104.5))
            y = float(np.clip(rng.normal(34.0, 9.0), 1.0, 67.0))
            dx, dy = 105.0 - x, y - 34.0
            distance = math.hypot(dx, dy)
            angle = math.atan2(7.32 * dx, dx * dx + dy * dy - 3.66 ** 2)

            body_part = str(rng.choice(BODY_PARTS, p=BODY_WEIGHTS))
            shot_type = str(rng.choice(SHOT_TYPES, p=SHOT_WEIGHTS))
            set_piece = shot_type != "open_play"
            p = true_xg(distance, angle, body_part, set_piece)
            goal = int(rng.uniform() < p)

            has_ff = bool(rng.uniform() < 0.75)
            freeze = _freeze_frame(rng, x, y) if has_ff else []
            has_kp = bool(rng.uniform() < 0.55)

            rows.append({
                "shot_id": f"fixture-{shot_id:05d}",
                "match_id": match_id,
                "competition_id": cid,
                "season_id": sid,
                "comp_season": f"{comp} {season}",
                "competition": comp,
                "season": season,
                "gender": gender,
                "team_id": 1 if rng.uniform() < 0.5 else 2,
                "team": str(home if rng.uniform() < 0.5 else away),
                "player_id": int(rng.integers(1, len(PLAYERS) + 1)),
                "player": str(rng.choice(PLAYERS)),
                "period": int(rng.integers(1, 3)),
                "minute": int(rng.integers(0, 95)),
                "second": int(rng.integers(0, 60)),
                "position": "Centre Forward",
                "x": round(x, 3),
                "y": round(y, 3),
                "shot_type": shot_type,
                "technique": str(rng.choice(TECHNIQUES, p=[0.78, 0.12, 0.07, 0.03])),
                "play_pattern": str(rng.choice(PATTERNS,
                                               p=[0.62, 0.13, 0.11, 0.09, 0.05])),
                "body_part": body_part,
                "outcome": "goal" if goal else "saved",
                "goal": goal,
                "under_pressure": bool(rng.uniform() < 0.42),
                "shot_statsbomb_xg": (round(float(np.clip(p * rng.normal(1.0, 0.12),
                                                          0.002, 0.98)), 4)
                                      if rng.uniform() < 0.9 else None),
                "shot_first_time": bool(rng.uniform() < 0.33),
                "shot_one_on_one": bool(rng.uniform() < 0.05),
                "shot_open_goal": bool(rng.uniform() < 0.02),
                "shot_aerial_won": bool(rng.uniform() < 0.07),
                "shot_deflected": bool(rng.uniform() < 0.05),
                "shot_follows_dribble": bool(rng.uniform() < 0.06),
                "shot_redirect": bool(rng.uniform() < 0.03),
                "shot_saved_to_post": False,
                "shot_saved_off_target": False,
                "has_key_pass": has_kp,
                "pass_height": "ground_pass" if has_kp else None,
                "pass_type": None,
                "pass_technique": None,
                "pass_length": round(float(rng.gamma(2.0, 7.0)), 2) if has_kp else None,
                "pass_cross": bool(has_kp and rng.uniform() < 0.3),
                "pass_cut_back": bool(has_kp and rng.uniform() < 0.1),
                "pass_through_ball": bool(has_kp and rng.uniform() < 0.12),
                "pass_switch": False,
                "pass_shot_assist": has_kp,
                "pass_goal_assist": bool(has_kp and goal),
                "freeze_frame": json.dumps(freeze, separators=(",", ":")),
                "has_freeze_frame": has_ff,
            })

    # One shot per line: smaller than pretty-printing and still diffable in review.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
    goals = sum(r["goal"] for r in rows)
    print(f"{len(rows)} shots, {goals} goals ({goals / len(rows):.1%}), "
          f"{len({r['match_id'] for r in rows})} matches -> {OUT}")
    return 0


def _freeze_frame(rng, shot_x: float, shot_y: float) -> list[dict]:
    """A plausible freeze frame: a keeper near his line plus a handful of outfielders."""
    players = [{
        "teammate": False,
        "position": "Goalkeeper",
        "location": [round(float(rng.uniform(101.0, 104.8)), 2),
                     round(float(np.clip(rng.normal(34.0, 2.0), 30.0, 38.0)), 2)],
    }]
    for _ in range(int(rng.integers(2, 8))):
        players.append({
            "teammate": bool(rng.uniform() < 0.35),
            "position": "Centre Back",
            "location": [round(float(np.clip(rng.normal(shot_x + 4.0, 5.0), 50.0, 105.0)), 2),
                         round(float(np.clip(rng.normal(shot_y, 7.0), 0.0, 68.0)), 2)],
        })
    return players


if __name__ == "__main__":
    raise SystemExit(main())
