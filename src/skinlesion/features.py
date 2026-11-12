"""Hand-crafted image features for the classical (non-deep) baseline."""

from pathlib import Path

import numpy as np
from PIL import Image


def color_histogram(path: Path, bins: int = 8) -> np.ndarray:
    """Normalised RGB histogram: `bins` values per channel, 3 * bins in total, summing to 1."""
    img = Image.open(path).convert("RGB").resize((150, 112))
    arr = np.asarray(img)
    hists = [np.histogram(arr[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
    hist = np.concatenate(hists).astype(np.float32)
    return hist / hist.sum()
