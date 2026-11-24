"""Scoring rules, calibration diagnostics and a match-level bootstrap.

All written out rather than imported, partly so the repo is self-contained and partly
because the definitions matter: the calibration slope is a logistic regression of the
outcome on the logit of the prediction, and the expected calibration error here uses
equal-count bins, not equal-width ones. Equal-width bins put 95% of shots in the first
bucket and then report a calibration error dominated by three shots in the last.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "bootstrap_by_match",
    "brier_score",
    "calibration_slope_intercept",
    "expected_calibration_error",
    "log_loss",
    "logit",
    "reliability_table",
    "roc_auc",
    "summarise",
]

#: Predictions are clipped to this before any log, so one confident miss cannot be
#: infinitely penalised and sink an otherwise sane model's log loss.
EPS = 1e-9


def _prep(y, p):
    y = np.asarray(y, dtype="float64").ravel()
    p = np.clip(np.asarray(p, dtype="float64").ravel(), EPS, 1.0 - EPS)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    return y, p


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def log_loss(y, p) -> float:
    """Mean negative log likelihood, in nats. Lower is better."""
    y, p = _prep(y, p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y, p) -> float:
    """Mean squared error of the probability. Lower is better."""
    y, p = _prep(y, p)
    return float(np.mean((p - y) ** 2))


def roc_auc(y, p) -> float:
    """Area under the ROC curve, as the rank statistic, with ties averaged.

    Equivalent to the probability that a randomly chosen goal was given a higher xG than
    a randomly chosen non-goal.
    """
    y, p = _prep(y, p)
    pos, neg = y.sum(), (1.0 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype="float64")
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def reliability_table(y, p, *, n_bins: int = 10, min_per_bin: int = 50):
    """Equal-count bins of predicted probability against the observed goal rate.

    Bins are merged forwards until each holds at least ``min_per_bin`` shots, because a
    reliability point built on eight shots is noise drawn as signal.
    """
    y, p = _prep(y, p)
    if len(p) == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    y, p = y[order], p[order]

    # Equal-count cut points, then walk forward and only close a bin once it is big
    # enough. Whatever is left at the end joins the final bin rather than forming a
    # short one of its own.
    cuts = [int(round(i * len(p) / n_bins)) for i in range(1, n_bins + 1)]
    bounds: list[tuple[int, int]] = []
    start = 0
    for end in cuts:
        # Never split a run of identical predictions: shots the model cannot tell apart
        # must share a bin, or the table invents a calibration gap out of sort order.
        end = _extend_past_ties(p, end)
        if end - start >= min_per_bin:
            bounds.append((start, end))
            start = end
    if start < len(p):
        if bounds:
            bounds[-1] = (bounds[-1][0], len(p))
        else:
            bounds.append((0, len(p)))

    return [
        {
            "n": int(end - start),
            "pred_mean": float(p[start:end].mean()),
            "pred_lo": float(p[start:end].min()),
            "pred_hi": float(p[start:end].max()),
            "observed": float(y[start:end].mean()),
            "goals": int(y[start:end].sum()),
        }
        for start, end in bounds
    ]


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function.

    ``1/(1+exp(-eta))`` overflows for eta below about -700, which a Newton step on a
    well-separated prediction reaches easily. Splitting on the sign keeps the exponent
    negative on both branches.
    """
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    exp_eta = np.exp(eta[~pos])
    out[~pos] = exp_eta / (1.0 + exp_eta)
    return out


def _extend_past_ties(sorted_p: np.ndarray, end: int) -> int:
    """Move a bin boundary forward until it no longer falls inside a tied run."""
    if end <= 0 or end >= len(sorted_p):
        return end
    value = sorted_p[end - 1]
    while end < len(sorted_p) and sorted_p[end] == value:
        end += 1
    return end


def expected_calibration_error(y, p, *, n_bins: int = 10, min_per_bin: int = 50) -> float:
    """Shot-weighted mean gap between predicted probability and observed goal rate."""
    rows = reliability_table(y, p, n_bins=n_bins, min_per_bin=min_per_bin)
    if not rows:
        return float("nan")
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["pred_mean"] - r["observed"]) for r in rows) / total)


def calibration_slope_intercept(y, p, *, max_iter: int = 100, tol: float = 1e-10):
    """Fit ``logit(P(goal)) = intercept + slope * logit(prediction)`` by Newton-Raphson.

    A perfectly calibrated model gives slope 1 and intercept 0. Slope below 1 means the
    predictions are too spread out (over-confident at both ends); a negative intercept
    means the model is systematically too high.
    """
    y, p = _prep(y, p)
    z = logit(p)
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        grad = X.T @ (y - mu)
        hess = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return float(beta[0]), float(beta[1])


def summarise(y, p, *, n_bins: int = 10, min_per_bin: int = 50) -> dict:
    """Every headline metric for one set of predictions."""
    intercept, slope = calibration_slope_intercept(y, p)
    y_arr, p_arr = _prep(y, p)
    return {
        "n_shots": int(len(y_arr)),
        "goals": int(y_arr.sum()),
        "mean_pred": float(p_arr.mean()),
        "log_loss": log_loss(y, p),
        "brier": brier_score(y, p),
        "auc": roc_auc(y, p),
        "ece": expected_calibration_error(y, p, n_bins=n_bins, min_per_bin=min_per_bin),
        "cal_intercept": intercept,
        "cal_slope": slope,
    }


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def bootstrap_by_match(y, p, match_ids, *, statistic=summarise, n_boot: int = 1000,
                       seed: int = 7, alpha: float = 0.05) -> dict:
    """Percentile confidence intervals from resampling whole matches with replacement.

    Resampling shots would treat the 25 shots in a match as 25 independent draws and
    give intervals roughly a third too narrow. Matches are the independent unit.
    """
    y = np.asarray(y, dtype="float64").ravel()
    p = np.asarray(p, dtype="float64").ravel()
    match_ids = np.asarray(match_ids).ravel()

    unique, inverse = np.unique(match_ids, return_inverse=True)
    by_match = [np.flatnonzero(inverse == i) for i in range(len(unique))]

    rng = np.random.default_rng(seed)
    draws: list[dict] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([by_match[i] for i in picks])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        draws.append(statistic(y[idx], p[idx]))

    point = statistic(y, p)
    out = {}
    for key, value in point.items():
        values = np.array([d[key] for d in draws], dtype="float64")
        values = values[np.isfinite(values)]
        if len(values) < 20:
            out[key] = {"value": value, "lo": float("nan"), "hi": float("nan")}
            continue
        out[key] = {
            "value": value,
            "lo": float(np.quantile(values, alpha / 2)),
            "hi": float(np.quantile(values, 1 - alpha / 2)),
        }
    out["_n_boot"] = len(draws)
    return out
