from __future__ import annotations

import numpy as np
import pytest

from rwd.masks import mask_to_polygons, polygons_to_yolo_lines, tile_starts, tile_windows


def test_tile_starts_cover_end():
    assert tile_starts(1280, 640, 0.2) == [0, 512, 640]
    assert tile_starts(500, 640, 0.2) == [0]
    assert tile_starts(640, 640, 0.0) == [0]


def test_tile_starts_rejects_bad_overlap():
    with pytest.raises(ValueError):
        tile_starts(100, 50, 1.0)


def test_tile_windows_cover_every_pixel():
    w, h = 1280, 960
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in tile_windows(w, h, 640, 0.2):
        assert x1 - x0 == 640 and y1 - y0 == 640
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_mask_to_polygons_drops_small_regions():
    m = np.zeros((100, 100), np.uint8)
    m[10:40, 10:40] = 255  # 900 px
    m[80:82, 80:82] = 255  # 4 px, noise
    polys = mask_to_polygons(m, min_area_px=20)
    assert len(polys) == 1
    xs, ys = polys[0][:, 0], polys[0][:, 1]
    assert xs.min() == 10 and xs.max() == 39 and ys.min() == 10 and ys.max() == 39


def test_yolo_lines_are_normalized():
    poly = np.array([[0, 0], [50, 0], [50, 25]], float)
    (line,) = polygons_to_yolo_lines([poly], width=100, height=50)
    assert line == "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"
