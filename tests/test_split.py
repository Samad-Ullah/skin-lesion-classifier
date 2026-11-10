import numpy as np
import pandas as pd
import pytest

from skinlesion.data.split import make_splits
from skinlesion.labels import CLASS_CODES


@pytest.fixture
def fake_meta():
    """Small fake metadata: 700 lesions, 1 to 3 images each, all 7 classes."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(700):
        dx = CLASS_CODES[i % 7]
        for j in range(rng.integers(1, 4)):
            rows.append({"lesion_id": f"L{i}", "image_id": f"I{i}_{j}", "dx": dx})
    return pd.DataFrame(rows)


def test_no_lesion_in_two_splits(fake_meta):
    s = make_splits(fake_meta)
    assert (s.groupby("lesion_id")["split"].nunique() == 1).all()


def test_every_image_exactly_once(fake_meta):
    s = make_splits(fake_meta)
    assert len(s) == len(fake_meta)
    assert s["image_id"].is_unique
    assert set(s["image_id"]) == set(fake_meta["image_id"])


def test_split_sizes(fake_meta):
    frac = make_splits(fake_meta)["split"].value_counts(normalize=True)
    assert 0.65 < frac["train"] < 0.78
    assert 0.10 < frac["val"] < 0.20
    assert 0.10 < frac["test"] < 0.20


def test_all_classes_in_every_split(fake_meta):
    for name, part in make_splits(fake_meta).groupby("split"):
        assert set(part["dx"]) == set(CLASS_CODES), name


def test_same_seed_same_split(fake_meta):
    first = make_splits(fake_meta, seed=1)
    second = make_splits(fake_meta, seed=1)
    pd.testing.assert_frame_equal(first, second)
