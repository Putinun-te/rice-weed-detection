"""Helpers that build tiny synthetic YOLO datasets for tests. No real images are used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rwd.yolo_io import write_data_yaml

SQUARE = "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"


def save_image(path: Path, seed: int, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def add_sample(root: Path, split: str, name: str, seed: int, label: str | None = SQUARE) -> None:
    save_image(root / "images" / split / f"{name}.png", seed)
    if label is not None:
        lp = root / "labels" / split / f"{name}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(label + ("\n" if label else ""))


@pytest.fixture
def clean_dataset(tmp_path):
    """train: 2 images, val: 1, test: 1. All distinct, one group per split."""
    root = tmp_path / "ds"
    add_sample(root, "train", "a", 1)
    add_sample(root, "train", "b", 2)
    add_sample(root, "val", "c", 3)
    add_sample(root, "test", "d", 4)
    write_data_yaml(root / "data.yaml", root, {0: "weedy_rice"}, ["train", "val", "test"])
    (root / "groups.csv").write_text(
        "image,group,source\n"
        "images/train/a.png,2024-06-02,a\n"
        "images/train/b.png,2024-06-04,b\n"
        "images/val/c.png,2024-09-30,c\n"
        "images/test/d.png,2025-01-15,d\n"
    )
    return root
