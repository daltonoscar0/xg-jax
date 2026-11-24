"""Match-level splits.

Shots from the same match are not independent: the same two teams, the same pitch, the
same referee, and in the open data the same analyst. Splitting by shot would put a
striker's first-half effort in train and his second-half effort in test, which inflates
every metric. Every split here is by ``match_id``, so a match lands entirely on one side.

A whole competition-season is held out on top of that. The random test set measures "how
well does this do on more of the same"; the held-out competition measures "how well does
this do somewhere it has never been", which is the question a club actually has.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_HOLDOUT", "DEFAULT_SEED", "Splits", "crossfit_folds", "make_splits"]

DEFAULT_SEED = 20260101
#: Held out entirely and never looked at during development.
DEFAULT_HOLDOUT = "Premier League 2015/2016"
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15


@dataclass
class Splits:
    seed: int
    holdout_competition: str
    train: list[int]
    val: list[int]
    test: list[int]
    holdout: list[int]

    def assign(self, match_ids) -> np.ndarray:
        """Label each match id ``train`` / ``val`` / ``test`` / ``holdout`` / ``unused``."""
        lookup = {}
        for name in ("train", "val", "test", "holdout"):
            lookup.update(dict.fromkeys(getattr(self, name), name))
        return np.array([lookup.get(int(m), "unused") for m in match_ids], dtype=object)

    def mask(self, shots: pd.DataFrame, name: str) -> np.ndarray:
        return self.assign(shots["match_id"].to_numpy()) == name

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Splits:
        return cls(**json.loads(Path(path).read_text()))

    def check_disjoint(self) -> None:
        names = ("train", "val", "test", "holdout")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = set(getattr(self, a)) & set(getattr(self, b))
                if overlap:
                    raise ValueError(f"{a} and {b} share {len(overlap)} match ids")


def make_splits(
    index: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    holdout_competition: str = DEFAULT_HOLDOUT,
) -> Splits:
    """70/15/15 by match, stratified by competition-season, with one competition held out.

    Stratifying matters because the open data is wildly unbalanced -- 18 La Liga seasons
    against one Indian Super League season. Without it a small competition can miss the
    test set entirely and the split stops being representative of the training mix.
    """
    index = index.drop_duplicates(subset="match_id")
    holdout = index.loc[index["comp_season"] == holdout_competition, "match_id"]
    holdout_ids = sorted(int(m) for m in holdout)
    pool = index.loc[index["comp_season"] != holdout_competition]

    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, group in pool.groupby("comp_season", sort=True):
        ids = np.sort(group["match_id"].to_numpy().astype("int64"))
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(TRAIN_FRACTION * n))
        n_val = int(round(VAL_FRACTION * n))
        # Tiny competitions: make sure train is never starved by rounding.
        n_train = min(max(n_train, 1), n)
        n_val = min(n_val, n - n_train)
        train += [int(m) for m in ids[:n_train]]
        val += [int(m) for m in ids[n_train:n_train + n_val]]
        test += [int(m) for m in ids[n_train + n_val:]]

    splits = Splits(seed=seed, holdout_competition=holdout_competition,
                    train=sorted(train), val=sorted(val), test=sorted(test),
                    holdout=holdout_ids)
    splits.check_disjoint()
    return splits


def crossfit_folds(match_ids, *, k: int = 5, seed: int = DEFAULT_SEED) -> dict[int, int]:
    """Assign every match to one of ``k`` folds, for out-of-fold scoring of every shot.

    The finishing analysis needs an xG for every shot in the data, and each one has to
    come from a model that never saw that match. K folds by match give exactly that.
    """
    ids = np.sort(np.unique(np.asarray(match_ids, dtype="int64")))
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(ids))
    return {int(ids[idx]): int(i % k) for i, idx in enumerate(order)}
