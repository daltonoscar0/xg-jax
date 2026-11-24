"""Training: the loop descends, recovers a known answer, and stays reproducible."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xg.features import FeatureStats, build_frame, design_matrix
from xg.metrics import log_loss
from xg.models import (
    MLP,
    LogisticRegression,
    TemperatureScaler,
    bce_with_logits,
    build_model,
    predict_proba,
)
from xg.train import TrainConfig, fit, fit_temperature, load_model, save_model


def test_bce_matches_the_definition():
    logits = jnp.array([2.0, -1.0, 0.0])
    y = jnp.array([1.0, 0.0, 1.0])
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    expected = -np.mean(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log(1 - p))
    assert float(bce_with_logits(logits, y)) == pytest.approx(expected, rel=1e-6)


def test_bce_survives_an_extreme_logit():
    """The stable form must not produce inf where a naive sigmoid-then-log would."""
    assert np.isfinite(float(bce_with_logits(jnp.array([-400.0]), jnp.array([1.0]))))
    assert np.isfinite(float(bce_with_logits(jnp.array([400.0]), jnp.array([0.0]))))


def test_logistic_starts_at_the_base_rate_of_a_half():
    model = LogisticRegression(5, jax.random.PRNGKey(0))
    assert float(model(jnp.ones(5))) == pytest.approx(0.0)


def test_training_reduces_loss_on_a_known_problem(synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    split = 3000
    config = TrainConfig(model="logistic", max_epochs=30, patience=30)
    result = fit(X[:split], y[:split], X[split:], y[split:], config, verbose=False)

    start = result.history[0]["train_loss"]
    end = result.history[-1]["train_loss"]
    assert end < start
    # And it must beat predicting the base rate, which is the bar for "learned anything".
    base = float(y[:split].mean())
    assert end < log_loss(y[:split], np.full(split, base))


def test_training_recovers_the_generating_coefficients(synthetic_logistic):
    X, y, true_weights, true_bias = synthetic_logistic
    split = 3000
    result = fit(X[:split], y[:split], X[split:], y[split:],
                 TrainConfig(model="logistic", max_epochs=120, patience=120, lr=0.05),
                 verbose=False)
    weights = np.asarray(result.model.weight)
    assert weights == pytest.approx(true_weights, abs=0.25)
    assert float(result.model.bias) == pytest.approx(true_bias, abs=0.25)


def test_early_stopping_keeps_the_best_epoch_not_the_last(synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    split = 3000
    result = fit(X[:split], y[:split], X[split:], y[split:],
                 TrainConfig(model="mlp", max_epochs=60, patience=5, width=32),
                 verbose=False)
    best = min(h["val_loss"] for h in result.history)
    assert result.best_val_loss == pytest.approx(best)
    assert result.best_epoch <= result.history[-1]["epoch"]


def test_training_is_reproducible_from_the_seed(synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    split = 3000
    kwargs = dict(max_epochs=12, patience=12, width=16)
    a = fit(X[:split], y[:split], X[split:], y[split:],
            TrainConfig(model="mlp", seed=5, **kwargs), verbose=False)
    b = fit(X[:split], y[:split], X[split:], y[split:],
            TrainConfig(model="mlp", seed=5, **kwargs), verbose=False)
    c = fit(X[:split], y[:split], X[split:], y[split:],
            TrainConfig(model="mlp", seed=6, **kwargs), verbose=False)
    assert a.best_val_loss == pytest.approx(b.best_val_loss)
    assert a.best_val_loss != pytest.approx(c.best_val_loss)


def test_mlp_has_the_requested_shape():
    model = MLP(7, jax.random.PRNGKey(0), width=16, depth=2)
    assert len(model.layers) == 3
    assert model.layers[0].weight.shape == (16, 7)
    assert model.layers[-1].weight.shape == (1, 16)
    assert float(model(jnp.ones(7))).__class__ is float


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("random_forest", 3, jax.random.PRNGKey(0))


def test_predictions_are_probabilities(synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    result = fit(X[:2000], y[:2000], X[2000:], y[2000:],
                 TrainConfig(model="logistic", max_epochs=5, patience=5), verbose=False)
    p = predict_proba(result.model, X)
    assert p.shape == (len(X),)
    assert ((p > 0.0) & (p < 1.0)).all()


def test_batched_prediction_matches_unbatched(synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    model = build_model("mlp", X.shape[1], jax.random.PRNGKey(1), width=8, depth=2)
    assert predict_proba(model, X, batch=97) == pytest.approx(
        predict_proba(model, X, batch=100000), abs=1e-6)


# --------------------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------------------


def test_temperature_of_one_leaves_logits_alone():
    scaler = TemperatureScaler(0.0)
    assert scaler.temperature == pytest.approx(1.0)
    assert np.asarray(scaler(jnp.array([1.5, -2.0]))) == pytest.approx([1.5, -2.0])


def test_fitting_temperature_corrects_a_deliberately_over_spread_model(synthetic_logistic):
    """Train on doubled features so the logits come out too extreme, then fix them."""
    X, y, _, _ = synthetic_logistic

    class Doubled:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, x):
            return 2.5 * self.inner(x)

    base = fit(X[:3000], y[:3000], X[3000:], y[3000:],
               TrainConfig(model="logistic", max_epochs=60, patience=60, lr=0.05),
               verbose=False).model
    over = Doubled(base)

    before = log_loss(y[3000:], predict_proba(over, X[3000:]))
    scaler = fit_temperature(over, X[3000:], y[3000:], steps=300)
    after = log_loss(y[3000:], predict_proba(over, X[3000:], scaler=scaler))
    assert after < before
    # It should land near 2.5, which is exactly the over-spreading introduced above.
    assert scaler.temperature == pytest.approx(2.5, rel=0.25)


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path, synthetic_logistic):
    X, y, _, _ = synthetic_logistic
    result = fit(X[:2000], y[:2000], X[2000:], y[2000:],
                 TrainConfig(model="mlp", max_epochs=6, patience=6, width=8), verbose=False)
    path = tmp_path / "mlp.eqx"
    save_model(path, result.model, result.config, {"note": "round trip"})

    loaded, config, meta = load_model(path, X.shape[1])
    assert config.model == "mlp"
    assert meta["note"] == "round trip"
    assert predict_proba(loaded, X[:50]) == pytest.approx(predict_proba(result.model, X[:50]))


# --------------------------------------------------------------------------------------
# end to end on the fixture
# --------------------------------------------------------------------------------------


def test_fixture_shots_train_end_to_end(fixture_shots):
    """The real feature builder into the real training loop, on committed fixture data."""
    raw = build_frame(fixture_shots)
    train = np.arange(len(raw)) % 4 != 0
    stats = FeatureStats.fit(raw.loc[train])
    X = design_matrix(raw, stats)
    y = fixture_shots["goal"].to_numpy(dtype="float32")

    result = fit(X[train], y[train], X[~train], y[~train],
                 TrainConfig(model="logistic", max_epochs=40, patience=40, lr=0.05),
                 verbose=False)
    p = predict_proba(result.model, X[~train])
    base = np.full((~train).sum(), float(y[train].mean()))
    assert log_loss(y[~train], p) < log_loss(y[~train], base)
