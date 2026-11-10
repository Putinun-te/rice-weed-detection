"""End-to-end test of the WeedyRice preparation on a fake raw folder (no real data)."""

import csv

import numpy as np
import pytest
import yaml
from PIL import Image

from rwd.prepare_weedyrice import load_config, main, prepare, to_group
from rwd.validate import validate

DATES = {"DJI_0001": "2024:06:02 10:00:00", "DJI_0002": "2024:06:04 10:00:00",
         "DJI_0003": "2024:09:30 10:00:00", "DJI_0004": "2025:01:15 10:00:00"}


def make_raw(root, with_weed=True):
    (root / "RGB").mkdir(parents=True)
    (root / "Masks").mkdir()
    (root / "Metadata").mkdir()
    rng = np.random.default_rng(0)
    for i, stem in enumerate(DATES):
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(root / "RGB" / f"{stem}.JPG")
        mask = np.zeros((96, 128), np.uint8)
        if with_weed:
            mask[10 + i : 40 + i, 20:60] = 255
        Image.fromarray(mask).save(root / "Masks" / f"{stem}.png")
    with open(root / "Metadata/image_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "DateTimeOriginal", "Altitude"])
        for stem, dt in DATES.items():
            w.writerow([f"{stem}.JPG", dt, "12"])


def write_cfg(tmp_path, **overrides):
    cfg = {
        "source_dir": "raw", "out_dir": "out", "rgb_dir": "RGB", "mask_dir": "Masks", "mask_name": "{stem}.png",
        "metadata": {"csv": "Metadata/image_metadata.csv", "filename_column": "FileName",
                     "group_column": "DateTimeOriginal", "group_format": "date", "keep_columns": ["Altitude"]},
        "split_by_group": {"train": ["2024-06-02", "2024-06-04"], "val": ["2024-09-30"], "test": ["2025-01-15"]},
        "tile": {"size": None, "overlap": 0.2},
    }
    cfg.update(overrides)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_to_group():
    assert to_group("2024:06:02 10:00:00", "date") == "2024-06-02"
    assert to_group(" site A ", "raw") == "site A"
    with pytest.raises(ValueError):
        to_group("no date", "date")


def test_prepare_whole_images_passes_validation(tmp_path):
    make_raw(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path))
    summary = prepare(cfg)
    assert summary["train_images"] == 2 and summary["val_images"] == 1 and summary["test_images"] == 1
    out = tmp_path / "out"
    report = validate(out / "data.yaml", "segment", out / "groups.csv")
    assert report["ok"], report["errors"]
    assert report["leakage"]["groups_per_split"]["test"] == ["2025-01-15"]
    rows = list(csv.DictReader(open(out / "groups.csv")))
    assert rows[0]["Altitude"] == "12"


def test_prepare_tiles_keep_source_in_one_split(tmp_path):
    make_raw(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path, tile={"size": 64, "overlap": 0.2}))
    summary = prepare(cfg)
    assert summary["train_images"] == 2 * 6  # 128x96 -> 3 x 2 tiles of 64
    report = validate(tmp_path / "out/data.yaml", "segment", tmp_path / "out/groups.csv")
    assert report["ok"], report["errors"]


def test_unassigned_group_fails(tmp_path):
    make_raw(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path, split_by_group={"train": ["2024-06-02"], "test": ["2025-01-15"]}))
    with pytest.raises(ValueError, match="groups not listed"):
        prepare(cfg)


def test_same_group_in_two_splits_fails(tmp_path):
    make_raw(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path, split_by_group={"train": ["2024-06-02"], "test": ["2024-06-02"]}))
    with pytest.raises(ValueError, match="two splits"):
        prepare(cfg)


def test_wrong_column_lists_real_columns(tmp_path):
    make_raw(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path))
    cfg["metadata"]["group_column"] = None
    with pytest.raises(ValueError, match="DateTimeOriginal"):
        prepare(cfg)


def test_all_empty_masks_fail(tmp_path):
    make_raw(tmp_path / "raw", with_weed=False)
    with pytest.raises(RuntimeError, match="no weed pixels"):
        prepare(load_config(write_cfg(tmp_path)))


def test_refuses_to_delete_foreign_folder(tmp_path):
    make_raw(tmp_path / "raw")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("mine")
    cfg = load_config(write_cfg(tmp_path))
    with pytest.raises(FileExistsError):
        prepare(cfg, overwrite=True)
    assert (tmp_path / "out" / "keep.txt").exists()


def test_cli_rebuild_with_overwrite(tmp_path):
    make_raw(tmp_path / "raw")
    cfg_path = write_cfg(tmp_path)
    assert main(["--config", str(cfg_path)]) == 0
    assert main(["--config", str(cfg_path), "--overwrite"]) == 0
    assert (tmp_path / "out/validation_report.json").exists()
