"""Tiling and mask-to-polygon conversion for segmentation datasets.

Tiling is done on the image and its mask together, before polygons are made,
so a polygon never has to be clipped at a tile border.
"""

from __future__ import annotations

import cv2
import numpy as np


def tile_starts(length: int, size: int, overlap: float) -> list[int]:
    """Start offsets along one axis. The last tile is aligned to the end, so no pixel is dropped."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1 - overlap))))
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def tile_windows(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) windows covering the whole image. Small images give a single window."""
    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in tile_starts(height, size, overlap)
        for x in tile_starts(width, size, overlap)
    ]


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 20.0, epsilon_px: float = 1.0) -> list[np.ndarray]:
    """Outer contours of a binary mask as (N, 2) float arrays in pixel coordinates.

    - Holes inside a weed patch are filled (RETR_EXTERNAL). Fine for patch detection.
    - Regions smaller than min_area_px are dropped as annotation noise.
    - epsilon_px simplifies the outline to keep label files small.
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        approx = cv2.approxPolyDP(c, epsilon_px, True) if epsilon_px > 0 else c
        pts = approx.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def polygons_to_yolo_lines(polygons: list[np.ndarray], width: int, height: int, class_id: int = 0) -> list[str]:
    lines = []
    for pts in polygons:
        norm = np.clip(pts / [width, height], 0.0, 1.0)
        lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in norm.flatten()))
    return lines
