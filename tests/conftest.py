"""Fixtures. No network, no StatsBomb data, nothing over a few thousand rows.

``tests/fixtures/shots.json`` is a hand-written set of shots in the shape
:func:`xg.data.shots_for_match` produces. It is small, synthetic and committed, so CI can
exercise the feature builder, the splits and the leakage guard without the 1.7 GB
download -- and without redistributing StatsBomb data, which their user agreement forbids.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_shots() -> pd.DataFrame:
    """The committed synthetic shot table, in the shots-frame schema."""
    return pd.DataFrame(json.loads((FIXTURES / "shots.json").read_text()))


@pytest.fixture(scope="session")
def fixture_index(fixture_shots: pd.DataFrame) -> pd.DataFrame:
    """The match index implied by the fixture shots."""
    return (fixture_shots[["match_id", "competition_id", "season_id", "comp_season",
                           "competition", "season", "gender"]]
            .drop_duplicates(subset="match_id")
            .reset_index(drop=True))


@pytest.fixture
def synthetic_logistic():
    """A dataset where the true model is known: ``logit p = -1 + 2*x0 - 1.5*x1``.

    Used to check that the training step actually descends and that it recovers something
    close to the generating coefficients, which a loop that silently fails to apply
    updates would not.
    """
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2)).astype("float32")
    logits = -1.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype("float32")
    return X, y, np.array([2.0, -1.5]), -1.0
