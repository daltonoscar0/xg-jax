"""Finishing over and under expected goals.

The question is "did this player score more than his chances were worth", and it has one
hard requirement: the xG for a shot must come from a model that never saw that shot's
match. A model scoring its own training data gives a striker credit for the goals it was
fitted on, which pushes every residual towards zero and makes the whole table meaningless.

So this module cross-fits. Matches are split into ``k`` folds; for each fold a model is
trained on the other ``k-1``, *including a fresh standardisation fit*, and used to score
the held-out fold. Every shot in the data then has an out-of-fold xG, and no shot's xG
owes anything to its own match.

The second half of the module is about uncertainty, and it is the more important half. A
striker with 60 shots has a standard error on goals-minus-xG of roughly 2.3 goals, so a
"+3 finisher" over one season is an unremarkable draw. The bootstrap here resamples the
player's own *matches*, which is the unit that is independent, and the plot greys out
every interval that covers zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureStats, baseline_matrix, design_matrix
from .models import predict_proba
from .pipeline import MODEL_SPECS, Prepared, prepare, results_dir
from .splits import DEFAULT_SEED, crossfit_folds
from .train import TrainConfig, fit

__all__ = ["MIN_SHOTS", "chelsea_tables", "crossfit_predictions", "finishing_table",
           "run_analysis"]

#: Below this a player's goals-minus-xG is dominated by shot noise to the point of being
#: uninformative. Thirty is the spec's floor and already generous: see the README.
MIN_SHOTS = 30


def crossfit_predictions(
    prepared: Prepared,
    *,
    model: str = "logistic",
    k: int = 5,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> np.ndarray:
    """An out-of-fold xG for every shot in ``prepared``.

    Returns probabilities aligned to ``prepared.shots``. Each fold refits the feature
    standardisation on its own training rows, so the imputation medians and the mean/SD
    constants are as out-of-sample as the model weights.
    """
    shots = prepared.shots
    match_ids = shots["match_id"].to_numpy()
    folds = crossfit_folds(match_ids, k=k, seed=seed)
    fold_of_shot = np.array([folds[int(m)] for m in match_ids])

    spec = MODEL_SPECS[model]
    build = baseline_matrix if spec["matrix"] == "baseline" else design_matrix
    out = np.full(len(shots), np.nan, dtype="float64")

    for fold in range(k):
        held = fold_of_shot == fold
        rest = ~held
        # A validation slice carved out of the training folds, by match, for early
        # stopping. Using the held fold for that would leak it into the model selection.
        rest_matches = np.unique(match_ids[rest])
        rng = np.random.default_rng(seed + 100 + fold)
        val_matches = set(rng.choice(rest_matches,
                                     size=max(1, int(0.15 * len(rest_matches))),
                                     replace=False).tolist())
        is_val = np.array([int(m) in val_matches for m in match_ids]) & rest
        is_train = rest & ~is_val

        stats = FeatureStats.fit(prepared.raw.loc[is_train])
        X = build(prepared.raw, stats)
        result = fit(X[is_train], prepared.y[is_train], X[is_val], prepared.y[is_val],
                     TrainConfig(model=model, seed=seed + fold), verbose=False)
        out[held] = predict_proba(result.model, X[held])
        if verbose:
            print(f"  fold {fold + 1}/{k}: trained on {int(is_train.sum()):,} shots, "
                  f"scored {int(held.sum()):,} (val log loss {result.best_val_loss:.5f})")

    if not np.isfinite(out).all():
        raise RuntimeError("cross-fitting left some shots unscored")
    return out


# --------------------------------------------------------------------------------------
# per-player finishing
# --------------------------------------------------------------------------------------


def finishing_table(
    shots: pd.DataFrame,
    xg: np.ndarray,
    *,
    min_shots: int = MIN_SHOTS,
    n_boot: int = 2000,
    seed: int = 23,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Goals minus out-of-fold xG per player, with a cluster bootstrap over matches.

    The bootstrap resamples the player's own matches with replacement. Two shots in the
    same match share an opponent, a game state and a defensive line, so the match is the
    independent unit; resampling shots would narrow every interval by roughly a fifth and
    turn noise into findings.
    """
    frame = pd.DataFrame({
        "player": shots["player"].to_numpy(),
        "player_id": shots["player_id"].to_numpy(),
        "team": shots["team"].to_numpy(),
        "match_id": shots["match_id"].to_numpy(),
        "goal": shots["goal"].to_numpy(dtype="float64"),
        "xg": np.asarray(xg, dtype="float64"),
    }).dropna(subset=["player"])

    counts = frame.groupby("player_id")["goal"].size()
    keep = counts.loc[counts >= min_shots].index
    frame = frame.loc[frame["player_id"].isin(keep)]
    if frame.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for pid, group in frame.groupby("player_id", sort=False):
        # Collapse to one row per match first: that is what gets resampled.
        per_match = group.groupby("match_id")[["goal", "xg"]].sum()
        goals_m = per_match["goal"].to_numpy()
        xg_m = per_match["xg"].to_numpy()
        n_matches = len(per_match)

        picks = rng.integers(0, n_matches, size=(n_boot, n_matches))
        draws = goals_m[picks].sum(axis=1) - xg_m[picks].sum(axis=1)

        total_goals = float(goals_m.sum())
        total_xg = float(xg_m.sum())
        rows.append({
            "player": group["player"].iloc[0],
            "player_id": int(pid),
            "teams": ", ".join(sorted(set(group["team"].dropna()))[:3]),
            "shots": int(len(group)),
            "matches": int(n_matches),
            "goals": int(total_goals),
            "xg": round(total_xg, 3),
            "goals_minus_xg": round(total_goals - total_xg, 3),
            "per_shot": round((total_goals - total_xg) / len(group), 4),
            "ci_lo": round(float(np.quantile(draws, alpha / 2)), 3),
            "ci_hi": round(float(np.quantile(draws, 1 - alpha / 2)), 3),
        })

    out = pd.DataFrame(rows)
    out["significant"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out.sort_values("goals_minus_xg", ascending=False).reset_index(drop=True)


def noise_floor(shots: pd.DataFrame, xg: np.ndarray, *, n_shots: int = 60,
                edge: float = 0.02) -> dict:
    """How big a goals-minus-xG a purely average finisher produces by luck.

    Each shot is an independent Bernoulli draw with its own probability, so the variance
    of goals minus xG over ``n`` shots is ``sum p(1-p)``. Evaluated at the mean shot in
    the data, this is the number that decides whether any row in the table means anything.

    ``shots_to_detect_edge`` inverts the same arithmetic: a player who is genuinely
    ``edge`` better per shot accumulates ``edge * n`` goals of signal against
    ``1.96 * sqrt(var * n)`` of noise, so the two cross at ``1.96^2 * var / edge^2``. At
    a two-percentage-point edge -- which would make someone one of the best finishers
    alive -- that is a career's worth of shots, and that is the honest headline of this
    whole analysis.
    """
    p = np.asarray(xg, dtype="float64")
    p = p[np.isfinite(p)]
    var_per_shot = float(np.mean(p * (1.0 - p)))
    sd = float(np.sqrt(var_per_shot * n_shots))
    return {
        "n_shots": n_shots,
        "mean_xg_per_shot": round(float(p.mean()), 4),
        "sd_goals_minus_xg": round(sd, 3),
        "95pct_band": f"+/- {1.96 * sd:.2f} goals",
        "edge_per_shot": edge,
        "shots_to_detect_edge": int(round((1.96 ** 2) * var_per_shot / (edge ** 2))),
    }


# --------------------------------------------------------------------------------------
# Chelsea, if the data has a Premier League season
# --------------------------------------------------------------------------------------


def chelsea_tables(shots: pd.DataFrame, xg: np.ndarray, *,
                   team: str = "Chelsea") -> dict[str, pd.DataFrame]:
    """Team and player xG for one club, per competition-season it appears in.

    Returns an empty dict when the club is not in the open data at all, and otherwise
    reports the coverage (how many of the club's matches are present) alongside, because
    a partial season's xG totals are not a season's xG totals.
    """
    frame = shots.assign(xg=np.asarray(xg, dtype="float64"))
    involved = frame.loc[frame["team"] == team]
    if involved.empty:
        return {}

    seasons = sorted(involved["comp_season"].dropna().unique())
    match_ids = frame.loc[frame["comp_season"].isin(seasons), "match_id"]
    club_matches = sorted(involved["match_id"].unique())

    team_rows = []
    for season in seasons:
        season_shots = frame.loc[frame["comp_season"] == season]
        club = season_shots.loc[season_shots["team"] == team]
        mids = set(club["match_id"].unique())
        against = season_shots.loc[season_shots["match_id"].isin(mids)
                                   & (season_shots["team"] != team)]
        team_rows.append({
            "comp_season": season,
            "matches_in_data": int(len(mids)),
            "goals_for": int(club["goal"].sum()),
            "xg_for": round(float(club["xg"].sum()), 2),
            "goals_against": int(against["goal"].sum()),
            "xg_against": round(float(against["xg"].sum()), 2),
            "xg_difference": round(float(club["xg"].sum() - against["xg"].sum()), 2),
            "shots_for": int(len(club)),
            "shots_against": int(len(against)),
        })

    players = (involved.groupby(["player", "comp_season"], as_index=False)
               .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum")))
    players["goals_minus_xg"] = (players["goals"] - players["xg"]).round(3)
    players["xg"] = players["xg"].round(3)
    players = players.sort_values(["comp_season", "xg"], ascending=[True, False])

    return {
        "team": pd.DataFrame(team_rows),
        "players": players.reset_index(drop=True),
        "coverage": pd.DataFrame([{
            "team": team,
            "comp_seasons": len(seasons),
            "club_matches_in_data": len(club_matches),
            "all_matches_in_those_seasons": int(match_ids.nunique()),
        }]),
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def run_analysis(
    prepared: Prepared | None = None,
    *,
    model: str = "logistic",
    k: int = 5,
    n_boot: int = 2000,
    out: Path | None = None,
    team: str = "Chelsea",
    **prepare_kwargs,
) -> dict[str, pd.DataFrame]:
    """Cross-fit, build the finishing table, write ``results/``."""
    from . import plots

    prepared = prepared or prepare(**prepare_kwargs)
    out = Path(out or results_dir())
    out.mkdir(parents=True, exist_ok=True)

    print(f"cross-fitting {model} across {k} folds by match")
    xg = crossfit_predictions(prepared, model=model, k=k)
    np.save(out / "crossfit_xg.npy", xg)

    table = finishing_table(prepared.shots, xg, n_boot=n_boot)
    table.to_csv(out / "finishing.csv", index=False)
    if not table.empty:
        plots.finishing_plot(table, out / "finishing.png")

    floor = pd.DataFrame([noise_floor(prepared.shots, xg, n_shots=n)
                          for n in (30, 60, 100, 300)])
    floor.to_csv(out / "finishing_noise_floor.csv", index=False)

    club = chelsea_tables(prepared.shots, xg, team=team)
    for name, frame in club.items():
        frame.to_csv(out / f"{team.lower()}_{name}.csv", index=False)

    total_xg = float(np.sum(xg))
    total_goals = float(prepared.shots["goal"].sum())
    pd.DataFrame([{
        "model": model,
        "folds": k,
        "shots_scored": int(len(xg)),
        "total_goals": int(total_goals),
        "total_crossfit_xg": round(total_xg, 1),
        # A calibrated model's xG over the whole dataset should land on the goal count.
        "ratio": round(total_goals / total_xg, 4),
        "players_with_min_shots": int(len(table)),
        "players_with_interval_clear_of_zero": int(table["significant"].sum())
        if not table.empty else 0,
    }]).to_csv(out / "finishing_meta.csv", index=False)

    print(f"{len(table)} players with >= {MIN_SHOTS} shots; "
          f"{int(table['significant'].sum()) if not table.empty else 0} with an interval "
          "clear of zero")
    return {"finishing": table, "noise_floor": floor, **club}
