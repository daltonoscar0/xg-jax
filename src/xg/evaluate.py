"""Out-of-sample evaluation: metrics, calibration, breakdowns and feature effects.

Three decisions worth stating, because they are what make the numbers comparable:

* **The reference row is scored on the same shots.** StatsBomb's own xG is missing on a
  minority of shots. Every headline row -- ours and theirs -- is therefore computed on the
  subset of the split where ``shot_statsbomb_xg`` is present, so the comparison is
  shot-for-shot. The full-split numbers for our models are reported alongside, labelled.
* **Intervals resample matches.** See :func:`xg.metrics.bootstrap_by_match`.
* **Nothing here refits anything.** Models arrive trained, the temperature arrives fitted
  on validation, and the test set and held-out competition are touched exactly once per
  run, at the end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .features import CONTINUOUS, column_names
from .metrics import bootstrap_by_match, reliability_table, summarise
from .models import TemperatureScaler, predict_proba
from .pipeline import MODEL_SPECS, Prepared, prepare, results_dir
from .train import load_model

__all__ = ["EVAL_SPLITS", "Scored", "breakdowns", "coefficient_table", "evaluate",
           "load_predictions", "partial_dependence"]

#: The two questions: "more of the same" and "somewhere it has never been".
EVAL_SPLITS = {
    "test": "Held-out matches (random 15%)",
    "holdout": "Held-out competition (never seen in development)",
}

#: Breakdown slices. Each is (label, boolean column on the shots frame).
BREAKDOWNS = (
    ("open play", lambda s: ~s["is_set_piece"]),
    ("set piece", lambda s: s["is_set_piece"]),
    ("foot", lambda s: s["body_part"].isin(["right_foot", "left_foot"])),
    ("head", lambda s: s["body_part"].eq("head")),
    ("with freeze frame", lambda s: s["has_freeze_frame"]),
    ("without freeze frame", lambda s: ~s["has_freeze_frame"]),
)

MODELS_DIR = "models"


def models_dir() -> Path:
    return results_dir().parent / MODELS_DIR


@dataclass
class Scored:
    """Every model's probabilities for every shot, plus the StatsBomb reference."""

    prepared: Prepared
    preds: dict[str, np.ndarray]
    histories: dict[str, list[dict]]
    temperature: float | None

    def names(self) -> list[str]:
        return list(self.preds)


def load_predictions(prepared: Prepared, *, directory: Path | None = None) -> Scored:
    """Score every trained model on every shot, once.

    Predicting the whole table up front and slicing afterwards guarantees the test rows
    and the holdout rows come from identical model state, and makes the breakdowns free.
    """
    directory = Path(directory or models_dir())
    preds: dict[str, np.ndarray] = {}
    histories: dict[str, list[dict]] = {}
    temperature: float | None = None

    for name in MODEL_SPECS:
        path = directory / f"{name}.eqx"
        if not path.exists():
            continue
        X = prepared.matrix(name)
        model, _, meta = load_model(path, X.shape[1])
        preds[name] = predict_proba(model, X)
        histories[name] = meta.get("history", [])

        log_t = meta.get("temperature_log_t")
        if name == "mlp" and log_t is not None:
            scaler = TemperatureScaler(log_t)
            preds["mlp_temp"] = predict_proba(model, X, scaler=scaler)
            temperature = scaler.temperature

    if not preds:
        raise FileNotFoundError(
            f"no trained models in {directory}; run `xg train --model logistic` and "
            "`xg train --model mlp` first"
        )

    sb = pd.to_numeric(prepared.shots["shot_statsbomb_xg"], errors="coerce").to_numpy()
    preds["statsbomb"] = sb
    return Scored(prepared=prepared, preds=preds, histories=histories,
                  temperature=temperature)


# --------------------------------------------------------------------------------------
# metric tables
# --------------------------------------------------------------------------------------


def _rows_for_split(scored: Scored, split: str, *, n_boot: int) -> list[dict]:
    """One row per model per shot set, with bootstrap intervals."""
    prepared = scored.prepared
    idx = prepared.idx(split)
    if len(idx) == 0:
        return []

    y = prepared.y[idx]
    match_ids = prepared.shots["match_id"].to_numpy()[idx]
    sb = scored.preds["statsbomb"][idx]
    comparable = np.isfinite(sb)

    rows = []
    for name, p_all in scored.preds.items():
        p = p_all[idx]
        sets = [("comparable", comparable)]
        if name != "statsbomb" and not comparable.all():
            sets.append(("all", np.ones(len(idx), dtype=bool)))
        for shot_set, mask in sets:
            if mask.sum() == 0 or not np.isfinite(p[mask]).all():
                continue
            stats = bootstrap_by_match(y[mask], p[mask], match_ids[mask], n_boot=n_boot)
            row = {"split": split, "model": name, "label": _label(name),
                   "shot_set": shot_set}
            for key, value in stats.items():
                if key.startswith("_"):
                    continue
                row[key] = round(value["value"], 6)
                row[f"{key}_lo"] = round(value["lo"], 6)
                row[f"{key}_hi"] = round(value["hi"], 6)
            row["n_boot"] = stats["_n_boot"]
            rows.append(row)
    return rows


def _label(name: str) -> str:
    if name in MODEL_SPECS:
        return MODEL_SPECS[name]["label"]
    return {"mlp_temp": "MLP + temperature scaling",
            "statsbomb": "StatsBomb xG (reference)"}[name]


def breakdowns(scored: Scored, split: str) -> pd.DataFrame:
    """Metrics on slices a coach would ask about, on the comparable shot set."""
    prepared = scored.prepared
    idx = prepared.idx(split)
    if len(idx) == 0:
        return pd.DataFrame()

    shots = prepared.shots.iloc[idx].copy()
    shots["is_set_piece"] = prepared.raw["is_set_piece"].to_numpy()[idx].astype(bool)
    shots["has_freeze_frame"] = shots["has_freeze_frame"].fillna(False).astype(bool)
    y = prepared.y[idx]
    comparable = np.isfinite(scored.preds["statsbomb"][idx])

    rows = []
    for label, predicate in BREAKDOWNS:
        mask = predicate(shots).to_numpy() & comparable
        if mask.sum() < 200:  # too few shots for a stable calibration error
            rows.append({"split": split, "slice": label, "model": None,
                         "n_shots": int(mask.sum()), "note": "too few shots to score"})
            continue
        for name, p_all in scored.preds.items():
            stats = summarise(y[mask], p_all[idx][mask])
            rows.append({"split": split, "slice": label, "model": name,
                         "label": _label(name),
                         **{k: round(v, 6) if isinstance(v, float) else v
                            for k, v in stats.items()}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# feature effects
# --------------------------------------------------------------------------------------


def coefficient_table(*, directory: Path | None = None, n_features: int | None = None,
                      prepared: Prepared | None = None):
    """Logistic coefficients in standardised units, largest effect first.

    Standardised, so a coefficient is "logit change per standard deviation of this
    feature" for the continuous ones and "logit change when the flag is set" for the
    binary ones. The two are not comparable with each other, so the kind is labelled.

    Columns that are constant across the training split are marked not identifiable.
    They are collinear with the intercept, so the fit can put any value on them that the
    intercept offsets, and weight decay decides the split. ``has_freeze_frame`` is the
    live example: once penalties are excluded, every remaining shot has a freeze frame,
    and its coefficient means nothing at all. Reporting it unmarked would invite a reader
    to interpret the largest number in the table as a finding.
    """
    directory = Path(directory or models_dir())
    path = directory / "logistic.eqx"
    if not path.exists():
        return pd.DataFrame()

    names = column_names()
    model, _, _ = load_model(path, n_features or len(names))
    weights = np.asarray(model.weight, dtype="float64")
    if len(weights) != len(names):
        names = [f"feature_{i}" for i in range(len(weights))]

    out = pd.DataFrame({
        "feature": names,
        "coefficient": np.round(weights, 5),
        "kind": ["per SD" if n in CONTINUOUS else "0/1" for n in names],
        "odds_ratio": np.round(np.exp(weights), 4),
        "train_share_or_sd": _spread(prepared, names),
        "note": _identifiability(prepared, names),
    })
    out["abs"] = out["coefficient"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs")
    intercept = pd.DataFrame([{"feature": "(intercept)",
                               "coefficient": round(float(model.bias), 5),
                               "kind": "logit", "odds_ratio": None,
                               "train_share_or_sd": None, "note": ""}])
    return pd.concat([intercept, out], ignore_index=True)


def _train_columns(prepared: Prepared | None, names: list[str]) -> np.ndarray | None:
    """The training slice of the design matrix, if a :class:`Prepared` was supplied."""
    if prepared is None or prepared.full.shape[1] != len(names):
        return None
    return prepared.full[prepared.idx("train")]


def _spread(prepared: Prepared | None, names: list[str]):
    """Mean of each 0/1 column and standard deviation of each continuous one."""
    train = _train_columns(prepared, names)
    if train is None:
        return [None] * len(names)
    return [round(float(train[:, i].std() if n in CONTINUOUS else train[:, i].mean()), 5)
            for i, n in enumerate(names)]


def _identifiability(prepared: Prepared | None, names: list[str]) -> list[str]:
    """Mark columns the training data cannot pin down."""
    train = _train_columns(prepared, names)
    if train is None:
        return [""] * len(names)
    out = []
    for i, _ in enumerate(names):
        column = train[:, i]
        if float(column.std()) == 0.0:
            out.append("constant in training data: not identifiable, "
                       "collinear with the intercept")
        elif float((column != 0).mean()) < 0.001:
            out.append("set on under 0.1% of training shots: treat as noise")
        else:
            out.append("")
    return out


def partial_dependence(scored: Scored, split: str = "test", *, n_grid: int = 40,
                       max_rows: int = 4000, seed: int = 11,
                       directory: Path | None = None) -> dict:
    """Partial dependence on distance and angle for every model with a matrix.

    The textbook definition: hold the real rows fixed, move one column across a grid,
    average the predictions. Sub-sampled to ``max_rows`` background rows because the
    full cross product is 40 x 20,000 forward passes per model for a curve that is
    visually identical.
    """
    prepared = scored.prepared
    idx = prepared.idx(split)
    if len(idx) == 0:
        return {}

    rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)

    directory = Path(directory or models_dir())
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name in MODEL_SPECS:
        path = directory / f"{name}.eqx"
        if not path.exists():
            continue
        X = prepared.matrix(name)
        model, _, _ = load_model(path, X.shape[1])
        columns = ("distance", "angle") if MODEL_SPECS[name]["matrix"] == "baseline" \
            else tuple(column_names())

        curve: dict[str, np.ndarray] = {}
        for feature in ("distance", "angle"):
            col = columns.index(feature)
            raw = prepared.raw[feature].to_numpy(dtype="float64")[idx]
            grid = np.linspace(np.quantile(raw, 0.01), np.quantile(raw, 0.99), n_grid)
            # The matrix is standardised; the grid is in metres and radians.
            standardised = (grid - prepared.stats.mean[feature]) / prepared.stats.std[feature]

            background = X[idx]
            means = []
            for value in standardised:
                patched = background.copy()
                patched[:, col] = value
                means.append(float(predict_proba(model, patched).mean()))
            curve[feature] = grid
            curve[f"{feature}_mean"] = np.array(means)
        curves[name] = curve
    return curves


def _partial_dependence_frame(curves: dict) -> pd.DataFrame:
    rows = []
    for name, curve in curves.items():
        for feature in ("distance", "angle"):
            if feature not in curve:
                continue
            for value, mean in zip(curve[feature], curve[f"{feature}_mean"], strict=True):
                rows.append({"model": name, "feature": feature,
                             "value": round(float(value), 5),
                             "mean_xg": round(float(mean), 6)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# the whole evaluation
# --------------------------------------------------------------------------------------


def evaluate(
    prepared: Prepared | None = None,
    *,
    n_boot: int = 1000,
    n_bins: int = 10,
    out: Path | None = None,
    directory: Path | None = None,
    **prepare_kwargs,
) -> dict[str, pd.DataFrame]:
    """Score everything, write ``results/``, and return the tables."""
    prepared = prepared or prepare(**prepare_kwargs)
    directory = Path(directory or models_dir())
    scored = load_predictions(prepared, directory=directory)
    out = Path(out or results_dir())
    out.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    breakdown_frames: list[pd.DataFrame] = []
    reliability_rows: list[dict] = []

    for split, title in EVAL_SPLITS.items():
        idx = prepared.idx(split)
        if len(idx) == 0:
            print(f"  {split}: no shots, skipping")
            continue
        print(f"  {split}: {len(idx):,} shots")
        metric_rows += _rows_for_split(scored, split, n_boot=n_boot)
        breakdown_frames.append(breakdowns(scored, split))

        y = prepared.y[idx]
        comparable = np.isfinite(scored.preds["statsbomb"][idx])
        tables = {}
        for name, p_all in scored.preds.items():
            rows = reliability_table(y[comparable], p_all[idx][comparable], n_bins=n_bins)
            tables[name] = rows
            for i, row in enumerate(rows):
                reliability_rows.append({"split": split, "model": name, "bin": i,
                                         **{k: round(v, 6) if isinstance(v, float) else v
                                            for k, v in row.items()}})
        plots.reliability_plot(tables, out / f"reliability_{split}.png",
                               title=f"{title}\n{int(comparable.sum()):,} shots with a "
                                     "StatsBomb xG to compare against")
        plots.calibration_curve(tables, out / f"reliability_{split}_log.png", title=title)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(out / "reliability.csv", index=False)
    if breakdown_frames:
        pd.concat(breakdown_frames, ignore_index=True).to_csv(out / "breakdowns.csv",
                                                              index=False)

    coefficients = coefficient_table(directory=directory, prepared=prepared)
    if not coefficients.empty:
        coefficients.to_csv(out / "coefficients.csv", index=False)

    curves = partial_dependence(scored, directory=directory)
    if curves:
        _partial_dependence_frame(curves).to_csv(out / "partial_dependence.csv", index=False)
        plots.partial_dependence_plot(curves, out / "partial_dependence.png")

    if any(scored.histories.values()):
        plots.training_curve(scored.histories, out / "training_curves.png")

    (out / "evaluation_meta.json").write_text(json.dumps({
        "n_boot": n_boot,
        "n_bins": n_bins,
        "temperature": scored.temperature,
        "models": scored.names(),
        "splits": {k: int(len(prepared.idx(k))) for k in
                   ("train", "val", "test", "holdout")},
        "statsbomb_xg_coverage": round(
            float(np.isfinite(scored.preds["statsbomb"]).mean()), 4),
    }, indent=2) + "\n")

    print(f"wrote {out}")
    return {"metrics": metrics, "coefficients": coefficients}


def headline(metrics: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """The README table: one row per model on the comparable shot set."""
    if metrics.empty:
        return metrics
    sub = metrics.loc[(metrics["split"] == split) & (metrics["shot_set"] == "comparable")]
    order = ["baseline", "logistic", "mlp", "mlp_temp", "statsbomb"]
    sub = sub.assign(_o=sub["model"].map({m: i for i, m in enumerate(order)}))
    return sub.sort_values("_o").drop(columns="_o").reset_index(drop=True)
