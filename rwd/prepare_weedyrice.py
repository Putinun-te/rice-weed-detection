"""Build a YOLO-seg dataset from WeedyRice-RGBMS-DB (RGB images + binary masks).

Steps:
1. Read the metadata CSV and give every RGB image a group (by default its capture date).
2. Put each image in train/val/test from the group -> split table in the config.
   Groups are never split by random image, so one date/site stays in one split.
3. Optionally resize, then tile image and mask together.
4. Turn each mask tile into polygons and write YOLO-seg labels.
5. Write data.yaml, groups.csv (for the leakage check) and run the validator.

Usage:
    python -m rwd.prepare_weedyrice --config configs/weedyrice.yaml --inspect   # look at the raw files first
    python -m rwd.prepare_weedyrice --config configs/weedyrice.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from rwd.masks import mask_to_polygons, polygons_to_yolo_lines, tile_windows
from rwd.validate import print_summary, validate
from rwd.yolo_io import IMAGE_EXTS, write_data_yaml

MARKER = ".rwd_prepared"
DATE_RE = re.compile(r"(\d{4})[-:/_](\d{2})[-:/_](\d{2})")


@dataclass
class Sample:
    rgb: Path
    mask: Path
    group: str
    split: str
    extra: dict


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    base = Path(path).parent
    for key in ("source_dir", "out_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (base / p).resolve()
    return cfg


def to_group(value: str, fmt: str) -> str:
    if fmt == "raw":
        return value.strip()
    if fmt == "date":
        m = DATE_RE.search(value)
        if not m:
            raise ValueError(f"cannot read a date from {value!r}")
        return "-".join(m.groups())
    raise ValueError(f"group_format must be 'date' or 'raw', got {fmt!r}")


def read_metadata(cfg: dict) -> dict[str, dict]:
    """Map lower-case file stem -> metadata row."""
    meta = cfg["metadata"]
    csv_path = cfg["source_dir"] / meta["csv"]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    columns = list(rows[0])
    for key in ("filename_column", "group_column"):
        col = meta.get(key)
        if not col or col not in columns:
            raise ValueError(f"metadata.{key}={col!r} is not a column of {csv_path.name}. Columns: {columns}")
    for col in meta.get("keep_columns") or []:
        if col not in columns:
            raise ValueError(f"metadata.keep_columns has {col!r}, not in columns {columns}")

    by_stem = {}
    for r in rows:
        stem = Path(r[meta["filename_column"]].strip().replace("\\", "/")).stem.lower()
        if stem in by_stem:
            raise ValueError(f"file {stem!r} appears twice in {csv_path.name}")
        by_stem[stem] = r
    return by_stem


def collect_samples(cfg: dict) -> list[Sample]:
    src = cfg["source_dir"]
    rgb_dir, mask_dir = src / cfg["rgb_dir"], src / cfg["mask_dir"]
    meta_cfg = cfg["metadata"]
    metadata = read_metadata(cfg)

    split_of_group = {}
    for split, groups in cfg["split_by_group"].items():
        for g in groups:
            if str(g) in split_of_group:
                raise ValueError(f"group {g!r} is listed in two splits")
            split_of_group[str(g)] = split

    samples, unassigned, no_meta = [], Counter(), []
    for rgb in sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        mask = mask_dir / cfg["mask_name"].format(stem=rgb.stem)
        if not mask.exists():
            raise FileNotFoundError(f"mask not found for {rgb.name}: expected {mask}")
        row = metadata.get(rgb.stem.lower())
        if row is None:
            no_meta.append(rgb.name)
            continue
        group = to_group(row[meta_cfg["group_column"]], meta_cfg.get("group_format", "date"))
        split = split_of_group.get(group)
        if split is None:
            unassigned[group] += 1
            continue
        extra = {c: row[c] for c in meta_cfg.get("keep_columns") or []}
        samples.append(Sample(rgb, mask, group, split, extra))

    if no_meta:
        raise ValueError(f"{len(no_meta)} RGB image(s) have no metadata row, e.g. {no_meta[:3]}")
    if unassigned and not cfg.get("drop_unassigned_groups", False):
        raise ValueError(
            f"groups not listed in split_by_group: {dict(unassigned)}. "
            "Add them to a split, or set drop_unassigned_groups: true."
        )
    if not samples:
        raise ValueError("no samples matched the split table")
    return samples


def prepare(cfg: dict, overwrite: bool = False) -> dict:
    out = cfg["out_dir"]
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty. Use --overwrite to rebuild it.")
        if not (out / MARKER).exists():
            raise FileExistsError(f"{out} was not created by this script ({MARKER} missing). Not deleting it.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("created by rwd.prepare_weedyrice\n")

    samples = collect_samples(cfg)
    tile = cfg.get("tile") or {}
    tile_size, overlap = tile.get("size"), tile.get("overlap", 0.0)
    resize_max = cfg.get("resize_max_side")
    threshold = cfg.get("mask_threshold", 127)
    min_area, eps = cfg.get("min_area_px", 20), cfg.get("epsilon_px", 1.0)
    max_bytes = float(cfg.get("max_output_gb", 5)) * 1024**3
    jpeg_quality = int(cfg.get("jpeg_quality", 95))

    written_bytes = 0
    manifest = []
    stats = Counter()
    for s in samples:
        image = cv2.imread(str(s.rgb), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(s.mask), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"cannot read {s.rgb} or {s.mask}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"size mismatch {s.rgb.name} {image.shape[:2]} vs mask {mask.shape[:2]}")
        changed = False
        if resize_max and max(image.shape[:2]) > resize_max:
            scale = resize_max / max(image.shape[:2])
            new_wh = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, new_wh, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, new_wh, interpolation=cv2.INTER_NEAREST)
            changed = True
        binary = (mask > threshold).astype(np.uint8)

        h, w = binary.shape
        windows = tile_windows(w, h, tile_size, overlap) if tile_size else [(0, 0, w, h)]
        for x0, y0, x1, y1 in windows:
            whole = (x0, y0, x1, y1) == (0, 0, w, h)
            name = s.rgb.stem if whole else f"{s.rgb.stem}_x{x0}_y{y0}"
            img_dir, lbl_dir = out / "images" / s.split, out / "labels" / s.split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            if whole and not changed:
                dst = img_dir / s.rgb.name
                shutil.copy2(s.rgb, dst)
            else:
                dst = img_dir / f"{name}.jpg"
                cv2.imwrite(str(dst), image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            polygons = mask_to_polygons(binary[y0:y1, x0:x1], min_area, eps)
            lines = polygons_to_yolo_lines(polygons, x1 - x0, y1 - y0, class_id=0)
            lbl = lbl_dir / f"{dst.stem}.txt"
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

            written_bytes += dst.stat().st_size + lbl.stat().st_size
            if written_bytes > max_bytes:
                raise RuntimeError(f"output passed max_output_gb={cfg.get('max_output_gb', 5)}; stopping")
            stats[f"{s.split}_images"] += 1
            stats[f"{s.split}_with_weed"] += bool(lines)
            stats[f"{s.split}_polygons"] += len(lines)
            manifest.append(
                {"image": f"images/{s.split}/{dst.name}", "group": s.group, "source": s.rgb.stem, "split": s.split, **s.extra}
            )

    if sum(v for k, v in stats.items() if k.endswith("_with_weed")) == 0:
        raise RuntimeError("no weed pixels found in any mask; check mask_threshold and mask files")

    splits = [sp for sp in ("train", "val", "test") if stats[f"{sp}_images"]]
    write_data_yaml(out / "data.yaml", out, {0: cfg.get("class_name", "weedy_rice")}, splits)
    with open(out / "groups.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    summary = {"output_mb": round(written_bytes / 1024**2, 1), "source_images": len(samples), **stats}
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def date_columns(csv_path: Path) -> dict[str, Counter]:
    """Columns where every non-empty value contains a date, with image counts per date."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    found = {}
    for col in rows[0] if rows else []:
        values = [r[col] for r in rows if r[col] and r[col].strip()]
        if values and all(DATE_RE.search(v) for v in values):
            found[col] = Counter(to_group(v, "date") for v in values)
    return found


def inspect(cfg: dict) -> None:
    """Print what is in the raw folder, to fill in the config after download."""
    src = cfg["source_dir"]
    print(f"source_dir: {src}")
    for child in sorted(src.iterdir()):
        n = sum(1 for _ in child.rglob("*") if _.is_file()) if child.is_dir() else 1
        print(f"  {child.name}{'/' if child.is_dir() else ''}  ({n} files)")
    csv_path = src / cfg["metadata"]["csv"]
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [r for _, r in zip(range(3), reader)]
        print(f"columns of {csv_path.name}: {reader.fieldnames}")
        for r in rows:
            print(f"  {r}")
        for col, counts in date_columns(csv_path).items():
            print(f"images per date in column {col!r}: {dict(sorted(counts.items()))}")
    else:
        print(f"metadata csv not found: {csv_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--inspect", action="store_true", help="only list raw files and metadata columns")
    ap.add_argument("--overwrite", action="store_true", help="rebuild out_dir if this script created it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.inspect:
        inspect(cfg)
        return 0

    summary = prepare(cfg, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    out = cfg["out_dir"]
    report = validate(out / "data.yaml", "segment", out / "groups.csv", cfg.get("near_dup_bits"))
    (out / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
