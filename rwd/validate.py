"""Validate a YOLO-format dataset before training.

Checks:
- per split: image count, missing and empty label files, bad label lines, instances per class
- every class has instances in train (the old course dataset declared a class with zero instances)
- exact duplicate images (same bytes) inside a split and across splits
- optional near-duplicate images across splits (perceptual dHash)
- optional leakage check: the same group (field/site/date) or source image in more than one split

Usage:
    python -m rwd.validate data.yaml --task segment --groups groups.csv --out report.json

groups.csv columns: image (path relative to the dataset root), group, and optional source.
Exit code 1 if any error is found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from rwd.yolo_io import label_path_for, list_images, load_data_yaml, parse_label_file


def validate(
    data_yaml: str | Path,
    task: str,
    groups_csv: str | Path | None = None,
    near_dup_max_distance: int | None = None,
) -> dict:
    cfg = load_data_yaml(data_yaml)
    num_classes = len(cfg.names)
    errors: list[str] = []
    warnings: list[str] = []
    splits_report: dict[str, dict] = {}
    images_by_split: dict[str, list[Path]] = {}

    for split, images_dir in cfg.splits.items():
        images = list_images(images_dir)
        images_by_split[split] = images
        splits_report[split] = _check_split(split, images, task, cfg.names, num_classes, errors, warnings)

    if "train" not in cfg.splits:
        errors.append("data.yaml has no train split")
    else:
        for cid, name in cfg.names.items():
            if splits_report["train"]["instances"].get(name, 0) == 0:
                errors.append(f"class '{name}' (id {cid}) has zero instances in train")
    for split in ("val", "test"):
        if split in splits_report:
            for name in cfg.names.values():
                if splits_report[split]["instances"].get(name, 0) == 0:
                    warnings.append(f"class '{name}' has zero instances in {split}")

    duplicates = _exact_duplicates(images_by_split, cfg.root)
    if duplicates["across_splits"]:
        errors.append(f"{len(duplicates['across_splits'])} exact duplicate image group(s) across splits")
    if duplicates["within_split"]:
        warnings.append(f"{len(duplicates['within_split'])} exact duplicate image group(s) inside a split")

    near = []
    if near_dup_max_distance is not None:
        near = _near_duplicates_across_splits(images_by_split, cfg.root, near_dup_max_distance)
        if near:
            warnings.append(
                f"{len(near)} near-duplicate pair(s) across splits (dHash distance <= {near_dup_max_distance}); "
                "check them by eye, field images can look alike"
            )

    leakage = None
    if groups_csv is not None:
        leakage = _leakage(groups_csv, images_by_split, cfg.root, errors, warnings)

    return {
        "data_yaml": str(Path(data_yaml).resolve()),
        "task": task,
        "names": cfg.names,
        "splits": splits_report,
        "duplicates": {**duplicates, "near_across_splits": near},
        "leakage": leakage,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def _check_split(split, images, task, names, num_classes, errors, warnings) -> dict:
    instances: Counter = Counter()
    images_with_class: Counter = Counter()
    missing, empty, bad_lines = 0, 0, 0
    bad_examples = []

    for img in images:
        label = label_path_for(img)
        if not label.exists():
            missing += 1
            continue
        lines, problems = parse_label_file(label, task, num_classes)
        if problems:
            bad_lines += len(problems)
            for p in problems[: max(0, 5 - len(bad_examples))]:
                bad_examples.append(f"{label.name}:{p.line_no}: {p.reason}")
        if not lines and not problems:
            empty += 1
        for cid in {ln.class_id for ln in lines}:
            images_with_class[names[cid]] += 1
        for ln in lines:
            instances[names[ln.class_id]] += 1

    if not images:
        errors.append(f"split '{split}' has no images")
    if bad_lines:
        errors.append(f"split '{split}' has {bad_lines} bad label line(s), e.g. {bad_examples}")
    if missing:
        warnings.append(f"split '{split}' has {missing} image(s) without a label file (treated as background)")

    return {
        "images": len(images),
        "missing_labels": missing,
        "empty_labels": empty,
        "bad_lines": bad_lines,
        "instances": {n: instances.get(n, 0) for n in names.values()},
        "images_with_class": {n: images_with_class.get(n, 0) for n in names.values()},
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _exact_duplicates(images_by_split: dict[str, list[Path]], root: Path) -> dict:
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, images in images_by_split.items():
        for img in images:
            digest = hashlib.sha1(img.read_bytes()).hexdigest()
            by_hash[digest].append((split, _rel(img, root)))

    within, across = [], []
    for group in by_hash.values():
        if len(group) < 2:
            continue
        paths = [p for _, p in group]
        (across if len({s for s, _ in group}) > 1 else within).append(paths)
    return {"within_split": within, "across_splits": across}


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: compare neighbouring pixels of a small grayscale thumbnail."""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _near_duplicates_across_splits(images_by_split, root, max_distance) -> list[dict]:
    hashed = {s: [(dhash(p), _rel(p, root)) for p in imgs] for s, imgs in images_by_split.items()}
    split_names = list(hashed)
    pairs = []
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            if not hashed[a] or not hashed[b]:
                continue
            hb = np.array([h for h, _ in hashed[b]], dtype=np.uint64)
            for ha, pa in hashed[a]:
                xor = np.bitwise_xor(hb, np.uint64(ha))
                dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
                for j in np.nonzero(dist <= max_distance)[0]:
                    pairs.append({"a": pa, "b": hashed[b][j][1], "distance": int(dist[j])})
    return pairs


def _leakage(groups_csv, images_by_split, root, errors, warnings) -> dict:
    with open(groups_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"image", "group"} <= set(rows[0]):
        raise ValueError(f"{groups_csv} needs columns 'image' and 'group', has {list(rows[0])}")
    group_of = {r["image"]: r["group"] for r in rows}
    source_of = {r["image"]: r.get("source") or "" for r in rows}

    group_splits: dict[str, set] = defaultdict(set)
    source_splits: dict[str, set] = defaultdict(set)
    without_group = 0
    for split, images in images_by_split.items():
        for img in images:
            rel = _rel(img, root)
            if rel not in group_of:
                without_group += 1
                continue
            group_splits[group_of[rel]].add(split)
            if source_of[rel]:
                source_splits[source_of[rel]].add(split)

    leaked_groups = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    leaked_sources = {g: sorted(s) for g, s in source_splits.items() if len(s) > 1}
    if leaked_groups:
        errors.append(f"group leakage: {len(leaked_groups)} group(s) appear in more than one split: {leaked_groups}")
    if leaked_sources:
        errors.append(f"source leakage: {len(leaked_sources)} source image(s) appear in more than one split")
    if without_group:
        errors.append(f"{without_group} image(s) have no group in {groups_csv}; leakage cannot be checked for them")

    return {
        "groups_per_split": {
            s: sorted({group_of[_rel(p, root)] for p in imgs if _rel(p, root) in group_of})
            for s, imgs in images_by_split.items()
        },
        "groups_in_multiple_splits": leaked_groups,
        "sources_in_multiple_splits": leaked_sources,
        "images_without_group": without_group,
    }


def print_summary(report: dict) -> None:
    names = list(report["names"].values())
    print(f"{'split':<6} {'images':>7} {'empty':>6} {'missing':>8} " + " ".join(f"{n:>14}" for n in names))
    for split, r in report["splits"].items():
        counts = " ".join(f"{r['instances'][n]:>14}" for n in names)
        print(f"{split:<6} {r['images']:>7} {r['empty_labels']:>6} {r['missing_labels']:>8} {counts}")
    if report["leakage"]:
        for split, groups in report["leakage"]["groups_per_split"].items():
            print(f"groups in {split}: {groups}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    print("OK" if report["ok"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_yaml")
    ap.add_argument("--task", choices=["detect", "segment"], required=True)
    ap.add_argument("--groups", help="CSV with image,group[,source] for the leakage check")
    ap.add_argument("--near-dup", type=int, metavar="BITS", help="also flag near-duplicates across splits")
    ap.add_argument("--out", help="write the full report as JSON")
    args = ap.parse_args(argv)

    report = validate(args.data_yaml, args.task, args.groups, args.near_dup)
    print_summary(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
