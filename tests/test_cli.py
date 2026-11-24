"""The command line, including ``xg shot`` scoring a single shot without any data."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from xg import cli
from xg.cli import app
from xg.features import FeatureStats, build_frame, design_matrix
from xg.train import TrainConfig, fit, save_model

runner = CliRunner()


@pytest.fixture
def installed_model(fixture_shots, tmp_path, monkeypatch):
    """A trained logistic model plus its feature statistics, where the CLI looks."""
    raw = build_frame(fixture_shots)
    stats = FeatureStats.fit(raw)
    X = design_matrix(raw, stats)
    y = fixture_shots["goal"].to_numpy(dtype="float32")
    split = int(0.8 * len(X))

    result = fit(X[:split], y[:split], X[split:], y[split:],
                 TrainConfig(model="logistic", max_epochs=40, patience=40, lr=0.05),
                 verbose=False)
    save_model(tmp_path / "logistic.eqx", result.model, result.config,
               {"open_data_commit": "deadbeef"})
    (tmp_path / "feature_stats.json").write_text(json.dumps({
        "columns": list(stats.columns), "mean": stats.mean, "std": stats.std,
        "median": stats.median,
    }))
    monkeypatch.setattr(cli, "models_dir", lambda: tmp_path)
    return tmp_path


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fetch", "train", "evaluate", "analyse", "shot"):
        assert command in result.output


@pytest.mark.parametrize("command", ["fetch", "train", "evaluate", "analyse", "shot"])
def test_each_command_has_its_own_help(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_train_rejects_an_unknown_model():
    result = runner.invoke(app, ["train", "--model", "transformer"])
    assert result.exit_code != 0
    assert "unknown model" in result.output


def test_shot_without_a_trained_model_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "models_dir", lambda: tmp_path)
    result = runner.invoke(app, ["shot", "--x", "108", "--y", "40"])
    assert result.exit_code != 0
    assert "xg train" in result.output


def test_shot_scores_a_penalty_spot_shot(installed_model):
    result = runner.invoke(app, ["shot", "--x", "108", "--y", "40"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert 0.0 < payload["xg"] < 1.0
    # StatsBomb (108, 40) is the penalty spot: 11 m out, dead centre.
    assert payload["distance_m"] == pytest.approx(11.0, abs=0.05)
    assert payload["angle_deg"] == pytest.approx(36.81, abs=0.1)
    assert payload["trained_on_open_data_commit"] == "deadbeef"


def test_a_shot_from_closer_in_is_worth_more(installed_model):
    near = json.loads(runner.invoke(app, ["shot", "--x", "114", "--y", "40"]).output)
    far = json.loads(runner.invoke(app, ["shot", "--x", "90", "--y", "40"]).output)
    assert near["xg"] > far["xg"]


def test_a_shot_from_a_tight_angle_is_worth_less_than_one_dead_centre(installed_model):
    central = json.loads(runner.invoke(app, ["shot", "--x", "110", "--y", "40"]).output)
    wide = json.loads(runner.invoke(app, ["shot", "--x", "110", "--y", "72"]).output)
    assert central["xg"] > wide["xg"]


def test_headers_are_worth_less_than_the_same_shot_on_the_foot(installed_model):
    foot = json.loads(runner.invoke(
        app, ["shot", "--x", "110", "--y", "40", "--body-part", "right_foot"]).output)
    head = json.loads(runner.invoke(
        app, ["shot", "--x", "110", "--y", "40", "--body-part", "head"]).output)
    assert head["xg"] < foot["xg"]


def test_canonical_coordinates_are_accepted_directly(installed_model):
    out = json.loads(runner.invoke(
        app, ["shot", "--x", "94", "--y", "34", "--canonical"]).output)
    assert out["canonical_x"] == pytest.approx(94.0)
    assert out["distance_m"] == pytest.approx(11.0, abs=0.01)


def test_shot_says_the_freeze_frame_was_imputed(installed_model):
    out = json.loads(runner.invoke(app, ["shot", "--x", "108", "--y", "40"]).output)
    assert "imputed" in out["freeze_frame"]


def test_shot_keeps_the_freeze_frame_indicator_on():
    """Off would be out of distribution: every non-penalty training shot has one.

    The indicator is constant in training, so its coefficient is arbitrary. Turning it
    off at scoring time does not make the estimate cautious, it adds that arbitrary
    value to the logit.
    """
    row = cli._shot_row(94.0, 34.0, body_part="right_foot", shot_type="open_play",
                        play_pattern="regular_play", technique="normal",
                        assist_type="none", first_time=False, one_on_one=False,
                        under_pressure=False)
    assert row["has_freeze_frame"].iloc[0] == 1.0
    # ...and the freeze-frame features themselves are left missing, to be imputed.
    assert row["nearest_defender_dist"].isna().all()
