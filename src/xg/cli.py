"""The ``xg`` command line.

    xg fetch                       download the StatsBomb open data
    xg train --model logistic      fit and save one model
    xg evaluate                    metrics, calibration, plots
    xg analyse                     cross-fit and build the finishing table
    xg shot --x 108 --y 36 ...     score a single shot

``train`` saves to ``models/<name>.eqx`` with its config, history and (for the MLP) the
validation-fitted temperature next to it, so ``evaluate`` never has to retrain anything
and cannot accidentally score a differently-configured model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from .data import default_root, open_data_commit
from .evaluate import evaluate as run_evaluate
from .evaluate import headline, models_dir
from .features import CATEGORICALS, column_names
from .pipeline import MODEL_SPECS
from .splits import DEFAULT_HOLDOUT, DEFAULT_SEED

HELP = """Shot-level expected goals in JAX, on StatsBomb open data.

Typical run: xg fetch, then xg train for each model, then xg evaluate and xg analyse.
Use xg shot to score one shot from the command line.
"""

app = typer.Typer(add_completion=False, help=HELP, no_args_is_help=True)

__all__ = ["app", "main"]


@app.command()
def fetch(
    commit: str = typer.Option(None, help="pin a specific statsbomb/open-data commit"),
    root: Path = typer.Option(None, help="where to put the data"),
    force: bool = typer.Option(False, help="re-download even if already present"),
    extract: bool = typer.Option(False, help="unpack the 12.8 GB of event JSON too"),
) -> None:
    """Download the StatsBomb open data and record the commit used."""
    from .fetch import fetch_open_data

    fetch_open_data(root or default_root(), commit=commit, force=force,
                    extract_events=extract)


@app.command()
def train(
    model: str = typer.Option("logistic", help="baseline | logistic | mlp"),
    seed: int = typer.Option(DEFAULT_SEED, help="seed for init, shuffling and splits"),
    epochs: int = typer.Option(200, help="maximum epochs before early stopping gives up"),
    lr: float = typer.Option(3e-3, help="AdamW learning rate"),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay, weights only"),
    batch_size: int = typer.Option(1024),
    width: int = typer.Option(64, help="MLP hidden width"),
    depth: int = typer.Option(2, help="MLP hidden layers"),
    holdout: str = typer.Option(DEFAULT_HOLDOUT, help="competition-season held out entirely"),
    refresh: bool = typer.Option(False, help="rebuild the shot table from the raw data"),
) -> None:
    """Fit one model on the training split, early stopping on validation log loss."""
    from .pipeline import prepare
    from .train import TrainConfig, fit, fit_temperature, save_model

    if model not in MODEL_SPECS:
        raise typer.BadParameter(f"unknown model {model!r}; "
                                 f"choose from {', '.join(MODEL_SPECS)}")

    prepared = prepare(seed=seed, holdout=holdout, refresh=refresh)
    X = prepared.matrix(model)
    train_idx, val_idx = prepared.idx("train"), prepared.idx("val")
    typer.echo(f"{model}: {len(train_idx):,} train shots, {len(val_idx):,} validation, "
               f"{X.shape[1]} features")

    config = TrainConfig(model=model, seed=seed, lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=epochs, width=width,
                         depth=depth)
    result = fit(X[train_idx], prepared.y[train_idx], X[val_idx], prepared.y[val_idx],
                 config)

    extra: dict = {"history": result.history, "best_epoch": result.best_epoch,
                   "best_val_loss": result.best_val_loss,
                   "n_features": int(X.shape[1]),
                   "n_train_shots": int(len(train_idx)),
                   "open_data_commit": open_data_commit(),
                   "holdout_competition": holdout}
    if model == "mlp":
        # Fitted on validation logits only; the model itself is frozen by this point.
        scaler = fit_temperature(result.model, X[val_idx], prepared.y[val_idx])
        extra["temperature_log_t"] = float(scaler.log_t)
        extra["temperature"] = scaler.temperature
        typer.echo(f"  temperature {scaler.temperature:.4f}")

    path = models_dir() / f"{model}.eqx"
    save_model(path, result.model, config, extra)
    # The standardisation constants travel with the models: `xg shot` has to apply the
    # exact same transform, and refitting them from the data at scoring time would make
    # a single-shot prediction depend on whatever data happened to be on disk.
    stats_path = models_dir() / "feature_stats.json"
    stats_path.write_text(json.dumps({
        "columns": list(prepared.stats.columns),
        "mean": prepared.stats.mean,
        "std": prepared.stats.std,
        "median": prepared.stats.median,
        "fitted_on_train_shots": int(len(train_idx)),
        "seed": seed,
        "holdout_competition": holdout,
    }, indent=2) + "\n")
    typer.echo(f"saved {path} (best val log loss {result.best_val_loss:.5f} "
               f"at epoch {result.best_epoch})")


@app.command()
def evaluate(
    n_boot: int = typer.Option(1000, help="bootstrap resamples, over matches"),
    n_bins: int = typer.Option(10, help="reliability bins, equal count"),
    seed: int = typer.Option(DEFAULT_SEED),
    holdout: str = typer.Option(DEFAULT_HOLDOUT),
) -> None:
    """Score every trained model on the test set and the held-out competition."""
    tables = run_evaluate(n_boot=n_boot, n_bins=n_bins, seed=seed, holdout=holdout)
    metrics = tables["metrics"]
    for split in ("test", "holdout"):
        rows = headline(metrics, split)
        if rows.empty:
            continue
        typer.echo(f"\n{split}:")
        typer.echo(rows[["label", "n_shots", "log_loss", "brier", "auc", "ece",
                         "cal_slope"]].to_string(index=False))


@app.command()
def analyse(
    model: str = typer.Option("logistic", help="which model to cross-fit"),
    folds: int = typer.Option(5, help="cross-fitting folds, by match"),
    n_boot: int = typer.Option(2000, help="bootstrap resamples per player"),
    team: str = typer.Option("Chelsea", help="club for the club-specific tables"),
    seed: int = typer.Option(DEFAULT_SEED),
    holdout: str = typer.Option(DEFAULT_HOLDOUT),
) -> None:
    """Who finished above and below their xG, out of fold, with intervals."""
    from .analysis import run_analysis

    tables = run_analysis(model=model, k=folds, n_boot=n_boot, team=team, seed=seed,
                          holdout=holdout)
    table = tables["finishing"]
    if table.empty:
        typer.echo("no player cleared the minimum shot count")
        return
    cols = ["player", "shots", "goals", "xg", "goals_minus_xg", "ci_lo", "ci_hi"]
    typer.echo("\nfurthest above xG:")
    typer.echo(table.head(10)[cols].to_string(index=False))
    typer.echo("\nfurthest below xG:")
    typer.echo(table.tail(10)[cols].to_string(index=False))


@app.command(name="analyze", hidden=True)
def analyze_alias(
    model: str = typer.Option("logistic"),
    folds: int = typer.Option(5),
    n_boot: int = typer.Option(2000),
    team: str = typer.Option("Chelsea"),
    seed: int = typer.Option(DEFAULT_SEED),
    holdout: str = typer.Option(DEFAULT_HOLDOUT),
) -> None:
    """American spelling of ``analyse``."""
    analyse(model=model, folds=folds, n_boot=n_boot, team=team, seed=seed, holdout=holdout)


@app.command()
def shot(
    x: float = typer.Option(..., help="StatsBomb x, 0-120 (120 is the goal line)"),
    y: float = typer.Option(..., help="StatsBomb y, 0-80 (40 is the goal centre)"),
    body_part: str = typer.Option("right_foot", help="right_foot | left_foot | head | other"),
    shot_type: str = typer.Option("open_play", help=" | ".join(CATEGORICALS["shot_type"])),
    play_pattern: str = typer.Option("regular_play",
                                     help=" | ".join(CATEGORICALS["play_pattern"])),
    technique: str = typer.Option("normal", help=" | ".join(CATEGORICALS["technique"])),
    assist_type: str = typer.Option("none", help=" | ".join(CATEGORICALS["assist_type"])),
    first_time: bool = typer.Option(False, help="struck without a touch to control it"),
    one_on_one: bool = typer.Option(False, help="clean through on the keeper"),
    under_pressure: bool = typer.Option(False, help="a defender closing the shot down"),
    model: str = typer.Option("logistic", help="baseline | logistic | mlp"),
    canonical: bool = typer.Option(False,
                                   help="x/y are already metres on a 105x68 pitch"),
) -> None:
    """Score one shot from the command line.

    Freeze-frame features are unknown from the command line, so each is imputed with its
    training-set median -- a shot with defenders in the typical places.

    ``has_freeze_frame`` is deliberately left *on*. It looks more honest to turn it off,
    but every non-penalty shot in the training data has a freeze frame, so the flag is
    constant there and a zero is a combination the model has never seen. Feeding it one
    does not produce a cautious estimate, it produces whatever arbitrary value the fit
    happened to park on a column it could not identify -- in this run, about +1.1 in the
    logit, which triples the xG of a penalty-spot shot. The indicator exists for data
    that has real gaps; on this data it has nothing to say.
    """
    from .geometry import statsbomb_to_canonical
    from .models import predict_proba
    from .train import load_model

    path = models_dir() / f"{model}.eqx"
    stats_path = models_dir() / "feature_stats.json"
    if not path.exists() or not stats_path.exists():
        raise typer.BadParameter(f"no trained {model} in {models_dir()}; run `xg train` first")

    if canonical:
        cx, cy = float(x), float(y)
    else:
        ax, ay = statsbomb_to_canonical([x], [y])
        cx, cy = float(ax[0]), float(ay[0])

    row = _shot_row(cx, cy, body_part=body_part, shot_type=shot_type,
                    play_pattern=play_pattern, technique=technique,
                    assist_type=assist_type, first_time=first_time,
                    one_on_one=one_on_one, under_pressure=under_pressure)

    from .features import FeatureStats, baseline_matrix, design_matrix

    stats = FeatureStats(**_load_stats(stats_path))
    build = baseline_matrix if MODEL_SPECS[model]["matrix"] == "baseline" else design_matrix
    X = build(row, stats)

    loaded, _, meta = load_model(path, X.shape[1])
    p = float(predict_proba(loaded, X)[0])

    from .geometry import goal_angle, goal_distance

    typer.echo(json.dumps({
        "xg": round(p, 4),
        "model": MODEL_SPECS[model]["label"],
        "canonical_x": round(cx, 2),
        "canonical_y": round(cy, 2),
        "distance_m": round(float(goal_distance(cx, cy)), 2),
        "angle_deg": round(float(np.degrees(goal_angle(cx, cy))), 2),
        "freeze_frame": "not supplied: features imputed with training medians",
        "trained_on_open_data_commit": meta.get("open_data_commit", "unknown"),
    }, indent=2))


def _shot_row(cx: float, cy: float, **kwargs) -> pd.DataFrame:
    """One raw feature row, built the same way :func:`xg.features.build_frame` does."""
    from .features import build_frame

    shots = pd.DataFrame([{
        "x": cx, "y": cy,
        "body_part": kwargs["body_part"],
        "shot_type": kwargs["shot_type"],
        "play_pattern": kwargs["play_pattern"],
        "technique": kwargs["technique"],
        "shot_first_time": kwargs["first_time"],
        "shot_one_on_one": kwargs["one_on_one"],
        "shot_open_goal": False,
        "shot_aerial_won": False,
        "shot_follows_dribble": False,
        "shot_redirect": False,
        "under_pressure": kwargs["under_pressure"],
        # On, not off: see the note in `shot`. An empty freeze_frame still leaves every
        # freeze-frame feature NaN, so they are median-imputed as intended.
        "has_freeze_frame": True,
        "has_key_pass": kwargs["assist_type"] != "none",
        "freeze_frame": "[]",
        "pass_length": np.nan,
        "pass_cross": kwargs["assist_type"] == "cross",
        "pass_cut_back": kwargs["assist_type"] == "cutback",
        "pass_through_ball": kwargs["assist_type"] == "through_ball",
    }])
    return build_frame(shots)


def _load_stats(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    return {"columns": tuple(payload.get("columns") or column_names()),
            "mean": payload["mean"], "std": payload["std"], "median": payload["median"]}


def main() -> None:
    app()


if __name__ == "__main__":
    main()
