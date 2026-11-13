"""Add a second, train-only dataset to a prepared YOLO-seg dataset as an extra class.

Used for CoFly-WeedDB (cotton field, other weed species, CC BY 4.0, see DATASET.md). Its weeds are
NOT weedy rice, so they get their own class (`other_weed`) and are never mixed into `weedy_rice`.
Val and test are copied unchanged from the base dataset, so results stay comparable.

    python -m rwd.add_extra --config configs/cofly.yaml [--overwrite]

Config keys (paths relative to the config file):
    base_dataset   prepared dataset folder (with data.yaml and groups.csv)
    extra_images   folder of extra images
    extra_masks    folder of class-id masks with the same file names (0 = background, >0 = weed)
    out_dir        new dataset folder
    extra_class    name of the new class
    extra_prefix   prefix for the copied file names
    extra_group    group label written to groups.csv
    extra_altitude altitude label written to groups.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from rwd.masks import mask_to_polygons, polygons_to_yolo_lines
from rwd.validate import print_summary, validate
from rwd.yolo_io import list_images, load_data_yaml, write_data_yaml

MARKER = ".rwd_prepared"
PATH_KEYS = ("base_dataset", "extra_images", "extra_masks", "out_dir")


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for k in PATH_KEYS:
        cfg[k] = (path.parent / cfg[k]).resolve()
    return cfg


def short_name(stem: str, prefix: str) -> str:
    """'ID_00048_UAV_dji...[Lat=..]_DATE_..' -> 'cofly_ID_00048'. Other names: unsafe characters replaced."""
    m = re.match(r"(ID_\d+)", stem)
    return prefix + (m.group(1) if m else re.sub(r"[^A-Za-z0-9_.-]", "_", stem))


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # no extra disk space when on the same file system
    except OSError:
        shutil.copy2(src, dst)


def build(cfg: dict, overwrite: bool = False) -> dict:
    out: Path = cfg["out_dir"]
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty. Use --overwrite to rebuild it.")
        if not (out / MARKER).exists():
            raise FileExistsError(f"{out} was not created by this script ({MARKER} missing). Not deleting it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("created by rwd.add_extra\n")

    base = load_data_yaml(cfg["base_dataset"] / "data.yaml")
    extra_id = max(base.names) + 1
    names = {**base.names, extra_id: cfg["extra_class"]}
    stats = {}

    # 1. base dataset, unchanged (images, labels, groups)
    for split, images_dir in base.splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        images = list_images(images_dir)
        for img in images:
            _link_or_copy(img, out / "images" / split / img.name)
            lbl = base.root / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                _link_or_copy(lbl, out / "labels" / split / lbl.name)
        stats[f"base_{split}_images"] = len(images)
    with open(base.root / "groups.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    fields = list(manifest[0])

    # 2. extra dataset: train only, every weed class merged into the one extra class
    min_area, eps = cfg.get("min_area_px", 20), cfg.get("epsilon_px", 1.0)
    n_img = n_poly = 0
    for img in list_images(cfg["extra_images"]):
        mask_path = cfg["extra_masks"] / f"{img.stem}.png"
        if not mask_path.exists():
            continue  # unlabelled image: skipped, an empty label would teach "no weed here"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            raise ValueError(f"{mask_path} is not a single-channel class-id mask")
        size = cv2.imread(str(img)).shape[:2]
        if size != mask.shape:
            raise ValueError(f"size mismatch {img.name} {size} vs mask {mask.shape}")
        h, w = mask.shape
        polygons = mask_to_polygons((mask > 0).astype(np.uint8), min_area, eps)
        lines = polygons_to_yolo_lines(polygons, w, h, class_id=extra_id)
        name = short_name(img.stem, cfg.get("extra_prefix", "extra_"))
        dst = out / "images" / "train" / f"{name}{img.suffix.lower()}"
        if dst.exists():
            raise ValueError(f"name clash for {img.name} -> {dst.name}")
        _link_or_copy(img, dst)
        (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        values = {"image": f"images/train/{dst.name}", "group": cfg["extra_group"], "source": img.stem,
                  "split": "train", "altitude": cfg.get("extra_altitude", "")}
        manifest.append({k: values.get(k, "") for k in fields})  # same columns as the base groups.csv
        n_img += 1
        n_poly += len(lines)
    if n_img == 0:
        raise RuntimeError("no extra image with a mask was found; check extra_images and extra_masks")
    stats.update(extra_train_images=n_img, extra_train_polygons=n_poly, extra_class_id=extra_id)

    write_data_yaml(out / "data.yaml", out, names, list(base.splits))
    with open(out / "groups.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (out / "prepare_summary.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(json.dumps(build(cfg, args.overwrite), indent=2))
    report = validate(cfg["out_dir"] / "data.yaml", "segment", cfg["out_dir"] / "groups.csv")
    (cfg["out_dir"] / "validation_report.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
