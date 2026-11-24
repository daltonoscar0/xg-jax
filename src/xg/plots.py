"""Figures for ``results/``.

Matplotlib with the default style and no seaborn, so the plots render identically in CI.
Every figure here is drawn from a table that is also written to CSV, so a reader can
check the numbers behind the picture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI, and none wanted on a laptop either
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["calibration_curve", "finishing_plot", "partial_dependence_plot",
           "reliability_plot", "training_curve"]

#: One colour per model, reused across every figure so the eye can follow a model.
COLOURS = {
    "baseline": "#9aa0a6",
    "logistic": "#1f77b4",
    "mlp": "#d62728",
    "mlp_temp": "#ff7f0e",
    "statsbomb": "#2ca02c",
}

MARKERS = {"baseline": "s", "logistic": "o", "mlp": "^", "mlp_temp": "v", "statsbomb": "D"}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def reliability_plot(tables: dict[str, list[dict]], path: Path, *, title: str) -> Path:
    """Predicted vs observed goal rate, one line per model, with Wilson bars.

    The bars are the uncertainty in the *observed* rate within a bin, which is what tells
    a reader whether a point sitting off the diagonal is a real miscalibration or just a
    bin with forty shots in it.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=160)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--",
            label="perfect calibration", zorder=1)

    hi = 0.05
    for name, rows in tables.items():
        if not rows:
            continue
        pred = np.array([r["pred_mean"] for r in rows])
        obs = np.array([r["observed"] for r in rows])
        n = np.array([r["n"] for r in rows], dtype="float64")
        lo_err, hi_err = _wilson(obs, n)
        hi = max(hi, float(pred.max()), float((obs + hi_err).max()))
        ax.errorbar(pred, obs, yerr=[lo_err, hi_err], fmt=MARKERS.get(name, "o"),
                    markersize=5, linewidth=1.3, capsize=2.5, elinewidth=0.9,
                    color=COLOURS.get(name), label=_label(name), zorder=3)

    top = min(1.0, hi * 1.08)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("mean predicted xG in bin")
    ax.set_ylabel("observed goal rate in bin")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    return _save(fig, path)


def _wilson(obs: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval, as distances below and above the observed rate.

    Normal-approximation bars go negative in the low bins, where an xG model spends most
    of its predictions, so Wilson is the right interval here rather than a refinement.
    """
    denom = 1.0 + z * z / n
    centre = (obs + z * z / (2 * n)) / denom
    half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / denom
    lo = np.clip(obs - np.clip(centre - half, 0.0, 1.0), 0.0, None)
    hi = np.clip(np.clip(centre + half, 0.0, 1.0) - obs, 0.0, None)
    return lo, hi


def calibration_curve(tables: dict[str, list[dict]], path: Path, *, title: str) -> Path:
    """Reliability on a log x-axis, which is where an xG model actually lives.

    Most shots are worth under 0.1, so the linear plot crushes the interesting region
    into the bottom-left corner. This is the same data, readable.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=160)
    for name, rows in tables.items():
        if not rows:
            continue
        pred = np.array([r["pred_mean"] for r in rows])
        obs = np.array([r["observed"] for r in rows])
        n = np.array([r["n"] for r in rows], dtype="float64")
        lo_err, hi_err = _wilson(obs, n)
        ax.errorbar(pred, obs, yerr=[lo_err, hi_err], fmt=MARKERS.get(name, "o"),
                    markersize=5, linewidth=1.3, capsize=2.5, elinewidth=0.9,
                    color=COLOURS.get(name), label=_label(name))
    lim = [0.004, 1.0]
    ax.plot(lim, lim, color="black", linewidth=1.0, linestyle="--",
            label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted xG in bin (log scale)")
    ax.set_ylabel("observed goal rate in bin (log scale)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    return _save(fig, path)


def partial_dependence_plot(curves: dict[str, dict[str, np.ndarray]], path: Path) -> Path:
    """Partial dependence on distance and angle, one panel each.

    Averaged over the real test rows rather than over a grid of made-up ones, so the
    curve is "what does this model do to the shots that exist", not an extrapolation into
    feature combinations no footballer has ever produced.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    panels = [("distance", "distance to goal centre (m)"),
              ("angle", "angle subtended by the posts (degrees)")]
    for ax, (key, xlabel) in zip(axes, panels, strict=True):
        for name, curve in curves.items():
            if key not in curve:
                continue
            grid, mean = curve[key], curve[f"{key}_mean"]
            x = np.degrees(grid) if key == "angle" else grid
            ax.plot(x, mean, linewidth=1.8, color=COLOURS.get(name), label=_label(name))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean predicted xG")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Partial dependence on the two geometric features", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def training_curve(histories: dict[str, list[dict]], path: Path) -> Path:
    """Train and validation log loss per epoch, with the early-stopping pick marked."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
    for name, history in histories.items():
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        val = [h["val_loss"] for h in history]
        train = [h["train_loss"] for h in history]
        colour = COLOURS.get(name)
        ax.plot(epochs, train, linewidth=1.0, alpha=0.45, color=colour)
        ax.plot(epochs, val, linewidth=1.7, color=colour, label=f"{_label(name)} (val)")
        best = int(np.argmin(val))
        ax.plot([epochs[best]], [val[best]], marker="*", markersize=11, color=colour)
    ax.set_xlabel("epoch")
    ax.set_ylabel("log loss (nats)")
    ax.set_title("Training: faint line train, solid line validation, star the epoch kept",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, path)


def finishing_plot(table, path: Path, *, top: int = 12) -> Path:
    """The players furthest above and below their xG, with bootstrap intervals."""
    ordered = table.sort_values("goals_minus_xg")
    picks = [*range(min(top, len(ordered))), *range(max(0, len(ordered) - top), len(ordered))]
    picks = sorted(set(picks))
    sub = ordered.iloc[picks]

    fig, ax = plt.subplots(figsize=(7.6, 0.33 * len(sub) + 1.9), dpi=160)
    y = np.arange(len(sub))
    centre = sub["goals_minus_xg"].to_numpy()
    lo = centre - sub["ci_lo"].to_numpy()
    hi = sub["ci_hi"].to_numpy() - centre
    # Only an interval clear of zero is evidence of anything; the rest is shot noise.
    clear = (sub["ci_lo"] > 0) | (sub["ci_hi"] < 0)
    colours = np.where(clear, np.where(centre > 0, "#1a7f37", "#b42318"), "#9aa0a6")

    ax.errorbar(centre, y, xerr=[lo, hi], fmt="none", ecolor="#c6c9cc", elinewidth=1.4,
                capsize=2.5, zorder=2)
    ax.scatter(centre, y, s=34, c=colours, zorder=3)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}  ({int(n)} shots)"
                        for p, n in zip(sub["player"], sub["shots"], strict=True)],
                       fontsize=8)
    ax.set_xlabel("goals minus out-of-fold xG (95% CI, bootstrap over matches)")
    ax.set_title("Finishing over and under expected goals\n"
                 "grey = interval covers zero, i.e. indistinguishable from average",
                 fontsize=10)
    _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def _label(name: str) -> str:
    return {"baseline": "Logistic (distance + angle)",
            "logistic": "Logistic (all features)",
            "mlp": "MLP (2x64)",
            "mlp_temp": "MLP + temperature",
            "statsbomb": "StatsBomb xG"}.get(name, name)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
