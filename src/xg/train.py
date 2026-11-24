"""Training: minibatch AdamW with early stopping on validation log loss.

The step function is one ``jax.value_and_grad`` under ``jax.jit``; everything around it
is ordinary Python so the loop stays readable. Early stopping restores the parameters
from the best validation epoch rather than keeping the last ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .models import TemperatureScaler, bce_with_logits, build_model, predict_logits

__all__ = ["TrainConfig", "TrainResult", "fit", "fit_temperature", "load_model", "save_model"]


@dataclass
class TrainConfig:
    model: str = "logistic"
    seed: int = 20260101
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 20
    width: int = 64
    depth: int = 2

    def kwargs(self) -> dict:
        return {"width": self.width, "depth": self.depth} if self.model == "mlp" else {}


@dataclass
class TrainResult:
    model: object
    config: TrainConfig
    best_epoch: int
    best_val_loss: float
    history: list[dict] = field(default_factory=list)


def fit(X_train, y_train, X_val, y_val, config: TrainConfig, *, verbose: bool = True):
    """Fit one model, stopping when validation log loss stops improving."""
    X_train = jnp.asarray(np.asarray(X_train, dtype="float32"))
    y_train = jnp.asarray(np.asarray(y_train, dtype="float32"))
    X_val = jnp.asarray(np.asarray(X_val, dtype="float32"))
    y_val = jnp.asarray(np.asarray(y_val, dtype="float32"))

    key = jax.random.PRNGKey(config.seed)
    init_key, shuffle_key = jax.random.split(key)
    model = build_model(config.model, X_train.shape[1], init_key, **config.kwargs())

    # Weight decay on the weights only; biases are not a capacity knob and shrinking
    # them just biases the predicted base rate towards 0.5.
    optim = optax.adamw(config.lr, weight_decay=config.weight_decay,
                        mask=jax.tree_util.tree_map(lambda p: p.ndim > 0, model))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        loss, grads = eqx.filter_value_and_grad(_loss)(model, xb, yb)
        updates, opt_state = optim.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def evaluate(model, X, y):
        return _loss(model, X, y)

    n = X_train.shape[0]
    best_model, best_loss, best_epoch = model, float("inf"), -1
    history: list[dict] = []

    for epoch in range(config.max_epochs):
        shuffle_key, epoch_key = jax.random.split(shuffle_key)
        order = jax.random.permutation(epoch_key, n)
        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            model, opt_state, _ = step(model, opt_state, X_train[idx], y_train[idx])

        val_loss = float(evaluate(model, X_val, y_val))
        train_loss = float(evaluate(model, X_train, y_train))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-7:
            best_model, best_loss, best_epoch = model, val_loss, epoch
        elif epoch - best_epoch >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(best {best_loss:.5f} at epoch {best_epoch})")
            break
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

    return TrainResult(model=best_model, config=config, best_epoch=best_epoch,
                       best_val_loss=best_loss, history=history)


def _loss(model, X, y):
    return bce_with_logits(jax.vmap(model)(X), y)


def fit_temperature(model, X_val, y_val, *, steps: int = 500, lr: float = 0.05):
    """Fit the one temperature parameter on validation logits, with the model frozen."""
    logits = jnp.asarray(predict_logits(model, X_val), dtype=jnp.float32)
    y = jnp.asarray(np.asarray(y_val, dtype="float32"))

    scaler = TemperatureScaler()
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(scaler, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(scaler, opt_state):
        loss, grads = eqx.filter_value_and_grad(
            lambda s: bce_with_logits(s(logits), y))(scaler)
        updates, opt_state = optim.update(grads, opt_state,
                                          eqx.filter(scaler, eqx.is_inexact_array))
        return eqx.apply_updates(scaler, updates), opt_state, loss

    for _ in range(steps):
        scaler, opt_state, _ = step(scaler, opt_state)
    return scaler


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------


def save_model(path: Path, model, config: TrainConfig, extra: dict | None = None) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    meta = {"config": asdict(config), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_model(path: Path, n_features: int):
    import json

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    config = TrainConfig(**meta["config"])
    skeleton = build_model(config.model, n_features, jax.random.PRNGKey(config.seed),
                           **config.kwargs())
    return eqx.tree_deserialise_leaves(path, skeleton), config, meta
