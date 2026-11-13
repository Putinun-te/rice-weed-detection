"""Synthetic data only: a tiny base dataset plus a fake extra dataset with class-id masks."""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from rwd.add_extra import build, short_name
from rwd.evaluate import label_mask
from rwd.yolo_io import load_data_yaml


def make_cfg(tmp_path, base):
    images, masks = tmp_path / "extra/images", tmp_path / "extra/labels_1d"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for i, value in enumerate((1, 3)):
        name = f"ID_0000{i}_UAV_x_[Lat=1.5,Lon=2.5,Alt=5.0]_DATE_03_07_2019.png"
        cv2.imwrite(str(images / name), np.full((40, 60, 3), 90, np.uint8))
        m = np.zeros((40, 60), np.uint8)
        m[10:30, 10:40] = value
        cv2.imwrite(str(masks / name), m)
    cv2.imwrite(str(images / "ID_00009_unlabelled.png"), np.full((40, 60, 3), 90, np.uint8))
    return {"base_dataset": base, "extra_images": images, "extra_masks": masks, "out_dir": tmp_path / "out",
            "extra_class": "other_weed", "extra_prefix": "cofly_", "extra_group": "cofly-x", "extra_altitude": "5m"}


def test_short_name():
    assert short_name("ID_00048_UAV_dji_[Lat=39.5,Lon=22.6]_DATE", "cofly_") == "cofly_ID_00048"
    assert short_name("a b[1]", "x_") == "x_a_b_1_"


def test_build_adds_second_class_to_train_only(tmp_path, clean_dataset):
    cfg = make_cfg(tmp_path, clean_dataset)
    base = load_data_yaml(clean_dataset / "data.yaml")
    stats = build(cfg)
    out = load_data_yaml(cfg["out_dir"] / "data.yaml")
    assert out.names == {0: "weedy_rice", 1: "other_weed"}
    assert stats["extra_train_images"] == 2 and stats["extra_train_polygons"] == 2  # unlabelled image skipped

    for split in ("val", "test"):  # unchanged
        a = sorted(p.name for p in (clean_dataset / "images" / split).iterdir())
        b = sorted(p.name for p in (cfg["out_dir"] / "images" / split).iterdir())
        assert a == b
        for lbl in (cfg["out_dir"] / "labels" / split).glob("*.txt"):
            assert all(line.startswith("0 ") for line in lbl.read_text().splitlines())

    extra = sorted((cfg["out_dir"] / "labels/train").glob("cofly_*.txt"))
    assert len(extra) == 2
    assert all(line.startswith("1 ") for p in extra for line in p.read_text().splitlines())
    n_base_train = len(list((clean_dataset / "images/train").iterdir()))
    assert len(list((cfg["out_dir"] / "images/train").iterdir())) == n_base_train + 2
    assert set(base.splits) == set(out.splits)

    rows = list(csv.DictReader(open(cfg["out_dir"] / "groups.csv")))
    assert all(r["image"].startswith("images/train/cofly_") for r in rows if r["group"] == "cofly-x")
    assert sum(r["group"] == "cofly-x" for r in rows) == 2


def test_extra_class_is_not_counted_as_weedy_rice(tmp_path, clean_dataset):
    cfg = make_cfg(tmp_path, clean_dataset)
    build(cfg)
    img = cfg["out_dir"] / "images/train/cofly_ID_00000.png"
    assert not label_mask(img, 40, 60).any()               # class 0 = weedy_rice: nothing
    assert label_mask(img, 40, 60, class_id=1)[20, 20]     # class 1 = other_weed


def test_build_refuses_non_empty_out(tmp_path, clean_dataset):
    cfg = make_cfg(tmp_path, clean_dataset)
    build(cfg)
    with pytest.raises(FileExistsError):
        build(cfg)
    build(cfg, overwrite=True)
