"""The evaluation and finishing-analysis paths, end to end on the fixture.

These run the real :func:`xg.evaluate.evaluate` and :func:`xg.analysis.run_analysis`
against the committed synthetic shots, with models trained into a temp directory. They
are slower than the unit tests but they are the only thing that would catch the
evaluation wiring breaking: a reference row scored on the wrong shots, a plot that cannot
be drawn, a results file that stops being written.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xg.analysis import chelsea_tables, finishing_table, noise_floor
from xg.evaluate import (
    breakdowns,
    coefficient_table,
    evaluate,
    headline,
    load_predictions,
    partial_dependence,
)
from xg.features import FeatureStats, baseline_matrix, build_frame, design_matrix
from xg.pipeline import Prepared
from xg.splits import make_splits
from xg.train import TrainConfig, fit, fit_temperature, save_model

HOLDOUT = "Invented Cup 2025/2026"


@pytest.fixture(scope="module")
def prepared(fixture_shots, fixture_index) -> Prepared:
    """A :class:`Prepared` built straight from the fixture, bypassing the data loader."""
    shots = fixture_shots.reset_index(drop=True)
    raw = build_frame(shots)
    splits = make_splits(fixture_index, holdout_competition=HOLDOUT)
    where = splits.assign(shots["match_id"].to_numpy())
    stats = FeatureStats.fit(raw.loc[where == "train"])
    return Prepared(shots=shots, raw=raw, splits=splits, stats=stats,
                    full=design_matrix(raw, stats), base=baseline_matrix(raw, stats),
                    y=shots["goal"].to_numpy(dtype="float32"), where=where)


@pytest.fixture(scope="module")
def trained(prepared, tmp_path_factory):
    """Both models trained small and saved, as ``xg train`` would leave them."""
    directory = tmp_path_factory.mktemp("models")
    train_idx, val_idx = prepared.idx("train"), prepared.idx("val")

    for name in ("baseline", "logistic", "mlp"):
        X = prepared.matrix(name)
        config = TrainConfig(model=name, max_epochs=25, patience=25, lr=0.05, width=16)
        result = fit(X[train_idx], prepared.y[train_idx], X[val_idx],
                     prepared.y[val_idx], config, verbose=False)
        extra = {"history": result.history, "n_features": int(X.shape[1])}
        if name == "mlp":
            scaler = fit_temperature(result.model, X[val_idx], prepared.y[val_idx],
                                     steps=100)
            extra["temperature_log_t"] = float(scaler.log_t)
        save_model(directory / f"{name}.eqx", result.model, config, extra)
    return directory


@pytest.fixture(scope="module")
def evaluated(prepared, trained, tmp_path_factory):
    """One evaluation run, shared: it writes a dozen files and draws five figures."""
    out = tmp_path_factory.mktemp("results")
    tables = evaluate(prepared, n_boot=80, out=out, directory=trained)
    return tables, out


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def test_every_model_and_the_reference_get_scored(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    assert set(scored.names()) == {"baseline", "logistic", "mlp", "mlp_temp", "statsbomb"}
    for name in ("baseline", "logistic", "mlp", "mlp_temp"):
        p = scored.preds[name]
        assert p.shape == (len(prepared.shots),)
        assert ((p > 0) & (p < 1)).all()


def test_temperature_scaling_actually_changes_the_predictions(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    assert not np.allclose(scored.preds["mlp"], scored.preds["mlp_temp"])
    assert scored.temperature is not None and scored.temperature > 0


def test_missing_models_directory_is_an_error(prepared, tmp_path):
    with pytest.raises(FileNotFoundError, match="no trained models"):
        load_predictions(prepared, directory=tmp_path / "nothing")


def test_the_reference_row_is_scored_on_the_same_shots_as_the_models(evaluated):
    """Every 'comparable' row in one split must report an identical shot count."""
    metrics = evaluated[0]["metrics"]
    for split in metrics["split"].unique():
        rows = metrics.loc[(metrics["split"] == split) &
                           (metrics["shot_set"] == "comparable")]
        assert rows["n_shots"].nunique() == 1, "rows scored on different shot sets"
        assert "statsbomb" in set(rows["model"])


def test_our_models_are_also_reported_on_the_full_split(evaluated):
    """The fixture withholds StatsBomb xG on ~10% of shots, as the real data does.

    So the comparable set is a strict subset, and our models get a second row covering
    every shot -- otherwise the headline table would quietly drop 10% of the test set.
    """
    metrics = evaluated[0]["metrics"]
    assert "all" in set(metrics["shot_set"])
    for _, group in metrics.groupby("split"):
        rows = group.loc[group["model"] == "logistic"].set_index("shot_set")
        assert rows.loc["all", "n_shots"] > rows.loc["comparable", "n_shots"]
    # The reference only ever appears on the comparable set: it has no value elsewhere.
    assert set(metrics.loc[metrics["model"] == "statsbomb", "shot_set"]) == {"comparable"}


def test_evaluate_writes_the_results_a_reader_is_promised(evaluated):
    _, tmp_path = evaluated
    for name in ("metrics.csv", "reliability.csv", "breakdowns.csv", "coefficients.csv",
                 "partial_dependence.csv", "evaluation_meta.json",
                 "reliability_test.png", "partial_dependence.png",
                 "training_curves.png"):
        assert (tmp_path / name).exists(), f"{name} was not written"

    meta = json.loads((tmp_path / "evaluation_meta.json").read_text())
    assert meta["n_boot"] == 80
    assert 0.0 < meta["statsbomb_xg_coverage"] <= 1.0


def test_bootstrap_intervals_bracket_the_point_estimates(evaluated):
    metrics = evaluated[0]["metrics"]
    for _, row in metrics.iterrows():
        for key in ("log_loss", "brier"):
            if np.isfinite(row[f"{key}_lo"]):
                assert row[f"{key}_lo"] <= row[key] <= row[f"{key}_hi"]


def test_headline_orders_the_models_and_keeps_the_reference_last(evaluated):
    metrics = evaluated[0]["metrics"]
    rows = headline(metrics, "test")
    assert list(rows["model"])[-1] == "statsbomb"
    assert list(rows["model"])[0] == "baseline"


def test_breakdowns_cover_the_slices_that_were_promised(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    frame = breakdowns(scored, "test")
    assert set(frame["slice"]) == {"open play", "set piece", "foot", "head",
                                   "with freeze frame", "without freeze frame"}


def test_coefficients_are_reported_for_every_design_column(prepared, trained):
    table = coefficient_table(directory=trained)
    assert table["feature"].iloc[0] == "(intercept)"
    assert len(table) == len(prepared.stats.columns) + 1
    assert not table["coefficient"].isna().any()


def test_partial_dependence_on_distance_falls_away(prepared, trained):
    """A model that did not learn "further is worse" is broken, fixture or not."""
    scored = load_predictions(prepared, directory=trained)
    curves = partial_dependence(scored, "test", n_grid=12, max_rows=200,
                                directory=trained)
    curve = curves["logistic"]
    assert curve["distance_mean"][:3].mean() > curve["distance_mean"][-3:].mean()


# --------------------------------------------------------------------------------------
# the finishing analysis
# --------------------------------------------------------------------------------------


def test_finishing_table_respects_the_minimum_shot_count(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    xg = scored.preds["logistic"]
    table = finishing_table(prepared.shots, xg, min_shots=10, n_boot=200)
    assert (table["shots"] >= 10).all()
    assert table["goals_minus_xg"].is_monotonic_decreasing


def test_goals_minus_xg_equals_goals_minus_the_summed_xg(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    xg = scored.preds["logistic"]
    table = finishing_table(prepared.shots, xg, min_shots=10, n_boot=50)
    row = table.iloc[0]
    own = prepared.shots["player_id"] == row["player_id"]
    assert row["goals"] == int(prepared.shots.loc[own, "goal"].sum())
    assert row["xg"] == pytest.approx(float(xg[own.to_numpy()].sum()), abs=1e-3)


def test_finishing_intervals_contain_the_point_estimate(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    table = finishing_table(prepared.shots, scored.preds["logistic"], min_shots=10,
                            n_boot=400)
    assert (table["ci_lo"] <= table["goals_minus_xg"]).all()
    assert (table["goals_minus_xg"] <= table["ci_hi"]).all()


def test_noise_floor_grows_with_the_square_root_of_shot_count(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    xg = scored.preds["logistic"]
    small = noise_floor(prepared.shots, xg, n_shots=50)
    large = noise_floor(prepared.shots, xg, n_shots=200)
    # Four times the shots, twice the standard deviation. The reported value is rounded
    # to three decimals, so the tolerance is that rounding and nothing more.
    assert large["sd_goals_minus_xg"] == pytest.approx(
        2.0 * small["sd_goals_minus_xg"], abs=0.002)


def test_club_tables_are_empty_for_a_club_not_in_the_data(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    assert chelsea_tables(prepared.shots, scored.preds["logistic"], team="Chelsea") == {}


def test_club_tables_report_for_and_against(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    out = chelsea_tables(prepared.shots, scored.preds["logistic"], team="Invented FC")
    assert set(out) == {"team", "players", "coverage"}
    assert (out["team"]["xg_for"] > 0).all()
    assert (out["team"]["shots_against"] >= 0).all()


def test_finishing_table_is_empty_when_nobody_clears_the_bar(prepared, trained):
    scored = load_predictions(prepared, directory=trained)
    out = finishing_table(prepared.shots, scored.preds["logistic"], min_shots=10_000)
    assert isinstance(out, pd.DataFrame)
    assert out.empty
