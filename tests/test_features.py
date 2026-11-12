import numpy as np
from PIL import Image

from skinlesion.features import color_histogram


def test_pure_red_image_puts_all_mass_in_extreme_bins(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (600, 450), color=(255, 0, 0)).save(path)

    h = color_histogram(path, bins=8)

    assert h.shape == (24,)
    assert h.sum() == np.float32(1.0)
    assert h[7] == np.float32(1 / 3)  # red channel, brightest bin
    assert h[8] == np.float32(1 / 3)  # green channel, darkest bin
    assert h[16] == np.float32(1 / 3)  # blue channel, darkest bin
