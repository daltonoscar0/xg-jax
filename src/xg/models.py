"""The models: logistic regression and a small MLP, both Equinox modules in JAX.

Each model maps a feature row to a single logit. Probabilities come from a sigmoid at the
edge, never inside the model, so the training loss can use the numerically stable
log-sum-exp form of binary cross entropy.

No class weighting and no resampling anywhere. Goals are about 10% of shots, which is not
rare enough to need either, and both would destroy calibration -- the whole point of an xG
model is that a 0.1 means one in ten, so an intervention that shifts the base rate is
worse than useless here.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["MLP", "LogisticRegression", "TemperatureScaler", "bce_with_logits", "build_model",
           "predict_proba"]


class LogisticRegression(eqx.Module):
    """One linear layer to a scalar logit."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self, n_features: int, key: jax.Array):
        # Zeros, not random: logistic regression is convex, the start does not matter,
        # and a zero start makes the fit exactly reproducible given the data order.
        del key
        self.weight = jnp.zeros((n_features,))
        self.bias = jnp.zeros(())

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.dot(self.weight, x) + self.bias


class MLP(eqx.Module):
    """Two hidden layers, ReLU, a scalar logit out.

    Deliberately small. With ~100k shots and ~40 features there is nothing to gain from
    depth: published work puts the ceiling for event-data xG not far above a good
    logistic model, and a bigger net just finds more ways to be over-confident.
    """

    layers: list
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(self, n_features: int, key: jax.Array, *, width: int = 64, depth: int = 2):
        keys = jax.random.split(key, depth + 1)
        sizes = [n_features] + [width] * depth + [1]
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys, strict=True)]
        self.width = width
        self.depth = depth

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).reshape(())


class TemperatureScaler(eqx.Module):
    """``p = sigmoid(logit / T)``, one parameter, fitted on the validation split.

    Temperature scaling can only fix the spread of the predictions, not a systematic
    offset, so the calibration intercept is reported alongside the slope to show whether
    the remaining error is the kind temperature can reach.
    """

    log_t: jax.Array

    def __init__(self, log_t: float = 0.0):
        self.log_t = jnp.asarray(log_t, dtype=jnp.float32)

    @property
    def temperature(self) -> float:
        return float(jnp.exp(self.log_t))

    def __call__(self, logits: jax.Array) -> jax.Array:
        return logits / jnp.exp(self.log_t)


def build_model(name: str, n_features: int, key: jax.Array, **kwargs):
    if name in {"logistic", "baseline"}:
        return LogisticRegression(n_features, key)
    if name == "mlp":
        return MLP(n_features, key, **kwargs)
    raise ValueError(f"unknown model {name!r}; use 'logistic', 'baseline' or 'mlp'")


def bce_with_logits(logits: jax.Array, y: jax.Array) -> jax.Array:
    """Mean binary cross entropy, computed from logits for numerical stability.

    ``max(z,0) - z*y + log(1 + exp(-|z|))`` never overflows, unlike taking a sigmoid
    first and then its log.
    """
    z = logits
    return jnp.mean(jnp.maximum(z, 0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z))))


@eqx.filter_jit
def forward(model, X: jax.Array) -> jax.Array:
    return jax.vmap(model)(X)


def predict_proba(model, X, *, scaler: TemperatureScaler | None = None, batch: int = 65536):
    """Probabilities for a feature matrix, in batches so a big test set fits in memory."""
    import numpy as np

    X = jnp.asarray(np.asarray(X, dtype="float32"))
    chunks = []
    for start in range(0, X.shape[0], batch):
        logits = forward(model, X[start:start + batch])
        if scaler is not None:
            logits = scaler(logits)
        chunks.append(jax.nn.sigmoid(logits))
    if not chunks:
        return np.zeros((0,), dtype="float64")
    return np.asarray(jnp.concatenate(chunks), dtype="float64")


def predict_logits(model, X, *, batch: int = 65536):
    import numpy as np

    X = jnp.asarray(np.asarray(X, dtype="float32"))
    chunks = [forward(model, X[s:s + batch]) for s in range(0, X.shape[0], batch)]
    if not chunks:
        return np.zeros((0,), dtype="float64")
    return np.asarray(jnp.concatenate(chunks), dtype="float64")
