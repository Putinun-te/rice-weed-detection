from __future__ import annotations

import numpy as np

from rwd.viz import COLOR, draw_labels, sample_images


def test_draw_labels_marks_polygon(clean_dataset):
    img = clean_dataset / "images/train/a.png"
    out = draw_labels(img, "segment", num_classes=1, alpha=1.0)
    assert out.shape == (48, 64, 3)
    # centre of the square label (x 0.1..0.3, y 0.1..0.3) is filled with the label colour
    assert tuple(out[int(48 * 0.2), int(64 * 0.2)]) == COLOR


def test_sample_images_is_repeatable(clean_dataset):
    a = sample_images(clean_dataset / "data.yaml", per_split=1, seed=1)
    b = sample_images(clean_dataset / "data.yaml", per_split=1, seed=1)
    assert a == b and set(a) == {"train", "val", "test"}
