"""Shared setup: shots in, split masks and design matrices out.

Both ``xg train`` and ``xg evaluate`` go through here so they cannot disagree about
which shots are in play, which matches are in which split, or what the standardisation
constants are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_shot_table, data_summary, default_root, match_index
from .features import FeatureStats, baseline_matrix, build_frame, design_matrix
from .splits import DEFAULT_HOLDOUT, DEFAULT_SEED, Splits, make_splits

__all__ = ["MODEL_SPECS", "Prepared", "penalty_summary", "prepare", "results_dir"]

#: Which matrix each model is trained on.
MODEL_SPECS = {
    "baseline": {"matrix": "baseline", "kind": "logistic",
                 "label": "Logistic (distance + angle)"},
    "logistic": {"matrix": "full", "kind": "logistic", "label": "Logistic (all features)"},
    "mlp": {"matrix": "full", "kind": "mlp", "label": "MLP (2x64)"},
}


def results_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


@dataclass
class Prepared:
    shots: pd.DataFrame
    raw: pd.DataFrame
    splits: Splits
    stats: FeatureStats
    full: np.ndarray
    base: np.ndarray
    y: np.ndarray
    where: np.ndarray

    def matrix(self, name: str) -> np.ndarray:
        return self.base if MODEL_SPECS[name]["matrix"] == "baseline" else self.full

    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.where == split)


def prepare(
    root: Path | None = None,
    *,
    refresh: bool = False,
    seed: int = DEFAULT_SEED,
    holdout: str = DEFAULT_HOLDOUT,
    save_splits: bool = True,
) -> Prepared:
    """Load every shot, drop the ones that are not modelled, and split by match."""
    root = Path(root or default_root())
    shots = build_shot_table(root, refresh=refresh)
    index = match_index(root)

    if save_splits:
        out = results_dir()
        out.mkdir(parents=True, exist_ok=True)
        data_summary(shots, index).to_csv(out / "data_summary.csv", index=False)
        penalty_summary(shots).to_csv(out / "penalties.csv", index=False)

    # Penalties are excluded: they are a near-constant ~0.78 with no useful variation,
    # and leaving them in flatters every metric by adding a large, trivially predictable
    # bucket. Their conversion rate is reported separately in results/penalties.csv.
    modelled = shots.loc[shots["shot_type"] != "penalty"].reset_index(drop=True)
    modelled = modelled.loc[modelled["x"].notna() & modelled["y"].notna()].reset_index(drop=True)

    splits = make_splits(index, seed=seed, holdout_competition=holdout)
    splits.check_disjoint()
    if save_splits:
        splits.save(results_dir() / "splits.json")

    raw = build_frame(modelled)
    where = splits.assign(modelled["match_id"].to_numpy())

    stats = FeatureStats.fit(raw.loc[where == "train"])
    return Prepared(
        shots=modelled,
        raw=raw,
        splits=splits,
        stats=stats,
        full=design_matrix(raw, stats),
        base=baseline_matrix(raw, stats),
        y=modelled["goal"].to_numpy(dtype="float32"),
        where=where,
    )


def penalty_summary(shots: pd.DataFrame) -> pd.DataFrame:
    """Penalty conversion, reported rather than modelled."""
    pens = shots.loc[shots["shot_type"] == "penalty"]
    if pens.empty:
        return pd.DataFrame([{"penalties": 0, "goals": 0, "conversion": float("nan")}])
    rows = [{
        "scope": "all open data",
        "penalties": int(len(pens)),
        "goals": int(pens["goal"].sum()),
        "conversion": round(float(pens["goal"].mean()), 4),
        "statsbomb_xg_mean": round(float(pens["shot_statsbomb_xg"].mean(skipna=True)), 4),
    }]
    for gender, group in pens.groupby("gender", dropna=True):
        rows.append({
            "scope": f"{gender}",
            "penalties": int(len(group)),
            "goals": int(group["goal"].sum()),
            "conversion": round(float(group["goal"].mean()), 4),
            "statsbomb_xg_mean": round(float(group["shot_statsbomb_xg"].mean(skipna=True)), 4),
        })
    return pd.DataFrame(rows)
