"""Splits are by match, and the holdout competition is really held out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xg.splits import Splits, crossfit_folds, make_splits


def test_no_match_appears_in_two_splits(fixture_index):
    splits = make_splits(fixture_index, holdout_competition="Invented Cup 2025/2026")
    sets = [set(splits.train), set(splits.val), set(splits.test), set(splits.holdout)]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not a & b


def test_every_match_is_assigned_somewhere(fixture_index):
    splits = make_splits(fixture_index, holdout_competition="Invented Cup 2025/2026")
    assigned = set(splits.train) | set(splits.val) | set(splits.test) | set(splits.holdout)
    assert assigned == set(fixture_index["match_id"].astype(int))


def test_every_shot_in_a_match_lands_on_the_same_side(fixture_shots, fixture_index):
    """The leakage this whole module exists to prevent."""
    splits = make_splits(fixture_index, holdout_competition="Invented Cup 2025/2026")
    where = splits.assign(fixture_shots["match_id"].to_numpy())
    per_match = pd.DataFrame({"match_id": fixture_shots["match_id"], "where": where})
    assert (per_match.groupby("match_id")["where"].nunique() == 1).all()


def test_holdout_is_exactly_the_named_competition(fixture_shots, fixture_index):
    splits = make_splits(fixture_index, holdout_competition="Invented Cup 2025/2026")
    where = splits.assign(fixture_shots["match_id"].to_numpy())
    held = fixture_shots.loc[where == "holdout", "comp_season"].unique()
    assert list(held) == ["Invented Cup 2025/2026"]
    # And nothing from that competition leaked into training.
    trained = set(fixture_shots.loc[where == "train", "comp_season"])
    assert "Invented Cup 2025/2026" not in trained


def test_split_is_stratified_so_each_competition_reaches_the_test_set(fixture_index):
    splits = make_splits(fixture_index, holdout_competition="Invented Cup 2025/2026")
    pool = fixture_index.loc[fixture_index["comp_season"] != "Invented Cup 2025/2026"]
    for _, group in pool.groupby("comp_season"):
        ids = set(group["match_id"].astype(int))
        assert ids & set(splits.train), "a competition missed training entirely"


def test_split_is_reproducible_from_the_seed(fixture_index):
    a = make_splits(fixture_index, seed=7)
    b = make_splits(fixture_index, seed=7)
    c = make_splits(fixture_index, seed=8)
    assert a.train == b.train
    assert a.train != c.train


def test_check_disjoint_catches_an_overlap():
    bad = Splits(seed=1, holdout_competition="x", train=[1, 2], val=[2], test=[3],
                 holdout=[4])
    with pytest.raises(ValueError, match="share"):
        bad.check_disjoint()


def test_save_and_load_round_trip(fixture_index, tmp_path):
    splits = make_splits(fixture_index)
    path = tmp_path / "splits.json"
    splits.save(path)
    assert Splits.load(path) == splits


def test_crossfit_covers_every_match_exactly_once(fixture_shots):
    ids = fixture_shots["match_id"].to_numpy()
    folds = crossfit_folds(ids, k=5)
    assert set(folds) == set(int(m) for m in ids)
    counts = np.bincount(list(folds.values()), minlength=5)
    # Folds within one match of each other: no fold is starved.
    assert counts.max() - counts.min() <= 1


def test_crossfit_assigns_a_whole_match_to_one_fold(fixture_shots):
    folds = crossfit_folds(fixture_shots["match_id"].to_numpy(), k=4)
    per_shot = [folds[int(m)] for m in fixture_shots["match_id"]]
    frame = pd.DataFrame({"match_id": fixture_shots["match_id"], "fold": per_shot})
    assert (frame.groupby("match_id")["fold"].nunique() == 1).all()
