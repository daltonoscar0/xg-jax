"""Shot-level expected goals in JAX, validated and calibrated on StatsBomb open data.

The public surface is small on purpose: :func:`xg.pipeline.prepare` loads and splits the
shots, :func:`xg.train.fit` fits one model, :func:`xg.evaluate.evaluate` scores everything.

Data is StatsBomb open data, used under the StatsBomb Public Data User Agreement. See the
README for the attribution StatsBomb require.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
