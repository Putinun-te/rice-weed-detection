"""End-to-end test of the WeedyRice preparation on a fake raw folder (no real data)."""

from __future__ import annotations

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


def test_date_columns_counts_images_per_date(tmp_path):
    from rwd.prepare_weedyrice import date_columns

    make_raw(tmp_path / "raw")
    found = date_columns(tmp_path / "raw/Metadata/image_metadata.csv")
    assert list(found) == ["DateTimeOriginal"]
    assert found["DateTimeOriginal"]["2024-06-02"] == 1


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


# --- Real WeedyRice layout: 5 metadata rows per capture (RGB + 4 MS bands), files on disk use
# --- the "standardized" names from filename_mapping.csv, masks share the RGB stem with .png.

STD = {stem: f"DJI_DateTime_{dt[:4]}_{dt[5:7]}_{dt[8:10]}_10_00_{i:04d}_lat_10.30_lon_105.26_alt_20.018m"
       for i, (stem, dt) in enumerate(DATES.items())}


def make_raw_mapped(root):
    (root / "RGB").mkdir(parents=True)
    (root / "Masks").mkdir()
    (root / "Metadata").mkdir()
    rng = np.random.default_rng(0)
    for i, stem in enumerate(DATES):
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(root / "RGB" / f"{STD[stem]}.JPG")
        mask = np.zeros((96, 128), np.uint8)
        mask[10 + i : 40 + i, 20:60] = 255
        Image.fromarray(mask).save(root / "Masks" / f"{STD[stem]}.png")
    bands = [("", "RGB", "FC6520", ".JPG"), ("_MS_G", "MS", "MSX01", ".TIF"), ("_MS_R", "MS", "MSX01", ".TIF"),
             ("_MS_RE", "MS", "MSX01", ".TIF"), ("_MS_NIR", "MS", "MSX01", ".TIF")]
    with open(root / "Metadata/image_metadata.csv", "w", newline="") as f, \
            open(root / "Metadata/filename_mapping.csv", "w", newline="") as g:
        w, m = csv.writer(f), csv.writer(g)
        w.writerow(["original_filename", "acquisition_date", "time", "altitude", "camera_model", "sensor_type", "band"])
        m.writerow(["original_filename", "standardized_filename"])
        for stem, dt in DATES.items():
            date = dt[:10].replace(":", "-")
            for suffix, sensor, cam, ext in bands:
                orig = f"{stem}{suffix}{ext}"
                std = f"{STD[stem]}{suffix.replace('_MS', '')}{ext}"
                w.writerow([orig, date, "10:00:00", "20.018m", cam, sensor, "" if sensor == "RGB" else suffix[4:]])
                m.writerow([orig, std])


def mapped_metadata(**extra):
    return {"csv": "Metadata/image_metadata.csv", "filename_column": "original_filename",
            "group_column": "acquisition_date", "group_format": "date", "keep_columns": ["altitude"],
            "row_filter": {"sensor_type": "RGB"},
            "filename_mapping": {"csv": "Metadata/filename_mapping.csv", "from_column": "original_filename",
                                 "to_column": "standardized_filename"}, **extra}


def test_mapping_and_row_filter_build_dataset(tmp_path):
    make_raw_mapped(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path, metadata=mapped_metadata()))
    summary = prepare(cfg)
    assert summary["train_images"] == 2 and summary["val_images"] == 1 and summary["test_images"] == 1
    out = tmp_path / "out"
    report = validate(out / "data.yaml", "segment", out / "groups.csv")
    assert report["ok"], report["errors"]
    rows = list(csv.DictReader(open(out / "groups.csv")))
    assert {r["group"] for r in rows} == {"2024-06-02", "2024-06-04", "2024-09-30", "2025-01-15"}
    assert rows[0]["altitude"] == "20.018m"
    assert rows[0]["source"].startswith("DJI_DateTime_")  # the on-disk (standardized) stem


def test_without_mapping_real_layout_fails_clearly(tmp_path):
    make_raw_mapped(tmp_path / "raw")
    meta = mapped_metadata()
    del meta["filename_mapping"]
    cfg = load_config(write_cfg(tmp_path, metadata=meta))
    with pytest.raises(ValueError, match="no metadata row"):
        prepare(cfg)


def test_row_filter_accepts_list_and_rejects_bad_column(tmp_path):
    make_raw_mapped(tmp_path / "raw")
    cfg = load_config(write_cfg(tmp_path, metadata=mapped_metadata(row_filter={"sensor_type": ["RGB"]})))
    assert prepare(cfg)["source_images"] == 4
    cfg = load_config(write_cfg(tmp_path, out_dir="out2", metadata=mapped_metadata(row_filter={"nope": "RGB"})))
    with pytest.raises(ValueError, match="row_filter column 'nope'"):
        prepare(cfg)
    cfg = load_config(write_cfg(tmp_path, out_dir="out3", metadata=mapped_metadata(row_filter={"sensor_type": "LIDAR"})))
    with pytest.raises(ValueError, match="keeps no row"):
        prepare(cfg)


def test_mapping_missing_entry_fails(tmp_path):
    make_raw_mapped(tmp_path / "raw")
    mp = tmp_path / "raw/Metadata/filename_mapping.csv"
    lines = [ln for ln in mp.read_text().splitlines() if not ln.startswith("DJI_0004.JPG,")]
    mp.write_text("\n".join(lines) + "\n")  # drop the RGB mapping row of one capture
    cfg = load_config(write_cfg(tmp_path, metadata=mapped_metadata()))
    with pytest.raises(ValueError, match="no entry in the filename mapping"):
        prepare(cfg)


def test_date_columns_with_row_filter_counts_rgb_only(tmp_path):
    from rwd.prepare_weedyrice import date_columns

    make_raw_mapped(tmp_path / "raw")
    p = tmp_path / "raw/Metadata/image_metadata.csv"
    assert date_columns(p)["acquisition_date"]["2024-06-02"] == 5
    assert date_columns(p, {"sensor_type": "RGB"})["acquisition_date"]["2024-06-02"] == 1


def test_inspect_prints_mapping_coverage(tmp_path, capsys):
    make_raw_mapped(tmp_path / "raw")
    assert main(["--config", str(write_cfg(tmp_path, metadata=mapped_metadata())), "--inspect"]) == 0
    out = capsys.readouterr().out
    assert "with row_filter" in out and "4 of 4 files in RGB/ are mapped" in out
