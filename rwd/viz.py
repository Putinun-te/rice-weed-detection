"""Draw YOLO labels on images for a quick visual check of the prepared dataset."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from rwd.yolo_io import label_path_for, list_images, load_data_yaml, parse_label_file

COLOR = (255, 0, 255)  # magenta, easy to see on green fields (RGB)


def draw_labels(image_path: Path, task: str, num_classes: int, alpha: float = 0.35) -> np.ndarray:
    """Return the image as RGB with label polygons (segment) or boxes (detect) drawn on it."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    label = label_path_for(image_path)
    if not label.exists():
        return rgb

    lines, _ = parse_label_file(label, task, num_classes)
    fill = rgb.copy()
    for ln in lines:
        c = np.array(ln.coords).reshape(-1, 2) * [w, h]
        if task == "detect":
            cx, cy, bw, bh = c.flatten()
            pts = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                            [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
        else:
            pts = c
        pts = pts.round().astype(np.int32)
        cv2.fillPoly(fill, [pts], COLOR)
        cv2.polylines(rgb, [pts], True, COLOR, 2)
    return cv2.addWeighted(fill, alpha, rgb, 1 - alpha, 0)


def sample_images(data_yaml: str | Path, per_split: int = 3, seed: int = 0, only_with_labels: bool = True) -> dict[str, list[Path]]:
    """Pick a few images per split with a fixed seed, preferring ones that have labels."""
    cfg = load_data_yaml(data_yaml)
    rng = random.Random(seed)
    picked = {}
    for split, d in cfg.splits.items():
        images = list_images(d)
        if only_with_labels:
            labelled = [p for p in images if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0]
            images = labelled or images
        picked[split] = rng.sample(images, min(per_split, len(images)))
    return picked
