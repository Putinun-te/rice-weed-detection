"""Evaluate a trained segmentation model at pixel level (Phase 4 tooling).

Why pixel level: the labels are free-form weedy-rice patches cut into instances by connected
components. A model can cover the same area but split or merge patches differently, and the
instance metric (mask mAP50) then counts both as wrong. The headline numbers are therefore
pixel precision, recall and IoU of the union of predicted masks against the label mask.
The instance metrics from Ultralytics are reported next to them, never hidden.

The test split is used once: `--split test` needs `--final`, takes exactly one `--conf`
(chosen on val beforehand) and refuses to run a second time.

Usage:
    python -m rwd.evaluate --weights runs/<name>/weights/best.pt --split val --conf 0.05 0.1 0.25 0.5
    python -m rwd.evaluate --weights runs/<name>/weights/best.pt --split test --conf 0.25 --final
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rwd.yolo_io import label_path_for, list_images, load_data_yaml, parse_label_file

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data/weedyrice_seg/data.yaml"


def rasterize(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Union of polygons (pixel coordinates, (N, 2)) as a boolean mask."""
    mask = np.zeros((height, width), np.uint8)
    for pts in polygons:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.asarray(pts).round().astype(np.int32)], 1)
    return mask.astype(bool)


def label_mask(image_path: Path, height: int, width: int, class_id: int = 0) -> np.ndarray:
    """Ground-truth mask of one class from the YOLO-seg label file of an image."""
    label = label_path_for(image_path)
    if not label.exists():
        return np.zeros((height, width), bool)
    lines, _ = parse_label_file(label, "segment", num_classes=10**6)
    return rasterize([np.array(ln.coords).reshape(-1, 2) * [width, height] for ln in lines if ln.class_id == class_id],
                     height, width)


def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"tp": int((pred & gt).sum()), "fp": int((pred & ~gt).sum()), "fn": int((~pred & gt).sum())}


def scores(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and IoU. None when undefined (nothing predicted / nothing labelled)."""
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
    }


def altitude_band(value: str | None, bands: tuple[int, ...] = (12, 20)) -> str:
    """'20.018m' -> '20m' (nearest nominal flight height). Unknown values -> 'unknown'."""
    try:
        alt = float(str(value).strip().rstrip("mM"))
    except (TypeError, ValueError):
        return "unknown"
    return f"{min(bands, key=lambda b: abs(b - alt))}m"


def summarize(rows: list[dict], by: str | None = None) -> dict:
    """Pool pixels over images (micro average). With `by`, one summary per value of that key."""
    if by is not None:
        keys = sorted({r[by] for r in rows})
        return {k: summarize([r for r in rows if r[by] == k]) for k in keys}
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    ious = [s for s in (scores(r["tp"], r["fp"], r["fn"])["iou"] for r in rows) if s is not None]
    out = {"images": len(rows), **scores(tp, fp, fn)}
    out["image_iou_median"] = float(np.median(ious)) if ious else None
    out["image_iou_min"] = float(min(ious)) if ious else None
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def error_overlay(image_bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB image: green = found weed, red = missed weed (FN), blue = false alarm (FP)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tint = rgb.copy()
    tint[pred & gt] = (0, 200, 0)
    tint[~pred & gt] = (255, 0, 0)
    tint[pred & ~gt] = (0, 80, 255)
    return cv2.addWeighted(tint, alpha, rgb, 1 - alpha, 0)


def predict_mask(model, image_path: Path, conf: float, imgsz, device, class_id: int = 0) -> np.ndarray:
    """Union of predicted masks of one class. Other classes (e.g. other_weed) are ignored."""
    r = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False, retina_masks=True,
                      classes=[class_id])[0]
    h, w = r.orig_shape
    return rasterize(list(r.masks.xy) if r.masks is not None else [], h, w)


def read_groups(groups_csv: Path) -> dict[str, dict]:
    if not groups_csv.exists():
        return {}
    with open(groups_csv, newline="") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


def evaluate(weights: Path, data_yaml: Path, split: str, confs: list[float], imgsz=1280, device=0,
             class_id: int = 0, tag: str = "") -> dict:
    """imgsz: an int, or [height, width] for an exported model with a fixed input size."""
    from ultralytics import YOLO

    data = load_data_yaml(data_yaml)
    if split not in data.splits:
        raise SystemExit(f"split {split!r} is not in {data_yaml}")
    images = list_images(data.splits[split])
    groups = read_groups(data.root / "groups.csv")
    model = YOLO(str(weights), task="segment")  # task is needed for exported models (.onnx, NCNN folder)

    per_conf, per_image = {}, {}
    for conf in confs:
        rows = []
        for p in images:
            pred = predict_mask(model, p, conf, imgsz, device, class_id)
            gt = label_mask(p, *pred.shape, class_id)
            meta = groups.get(p.relative_to(data.root).as_posix(), {})
            rows.append({"image": p.name, "group": meta.get("group", "unknown"),
                         "altitude": altitude_band(meta.get("altitude")), **pixel_counts(pred, gt)})
        key = f"{conf:g}"
        per_image[key] = rows
        per_conf[key] = {"all": summarize(rows), "by_altitude": summarize(rows, "altitude"), "by_group": summarize(rows, "group")}

    val = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=False, verbose=False,
                    classes=[class_id],
                    project=str(weights.resolve().parent.parent), name=f"eval_{split}{tag}_instance", exist_ok=True)
    instance = {k: round(float(v), 4) for k, v in val.results_dict.items() if "(M)" in k}
    return {"weights": str(weights), "data": str(data_yaml), "split": split, "imgsz": imgsz, "images": len(images),
            "class": data.names[class_id],
            "headline": "pixel precision / recall / IoU (union of predicted masks vs label mask)",
            "pixel": per_conf, "instance_default_conf": instance, "per_image": per_image}


def _fmt(value) -> str:
    """Table cell: 'n/a' when a score is undefined (e.g. nothing predicted at this conf)."""
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280],
                    help="one number, or height width for an exported model (e.g. 960 1280)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. onnx -> eval_val_onnx.json (val only)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--final", action="store_true", help="required for --split test (used once)")
    args = ap.parse_args(argv)

    run_dir = args.weights.resolve().parent.parent
    tag = f"_{args.tag}" if args.tag else ""
    out = run_dir / f"eval_{args.split}{tag}.json"
    if args.split == "test":
        if args.tag:
            raise SystemExit("--tag is for val only: the test split is used once per run.")
        if not args.final:
            raise SystemExit("The test split is used once, at the end, with the owner's OK. Add --final to confirm.")
        if len(args.conf) != 1:
            raise SystemExit("On test use exactly one --conf, chosen on val beforehand.")
        if out.exists():
            raise SystemExit(f"{out} exists: the test split was already used for this run. Not running again.")

    device = args.device if args.device == "cpu" else int(args.device)
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    report = evaluate(args.weights, args.data, args.split, args.conf, imgsz, device, tag=tag)
    out.write_text(json.dumps(report, indent=2))
    print(f"split: {args.split} ({report['images']} images)" + ("" if args.split == "test" else "  [model selection only, not publishable]"))
    print(f"{'conf':>6} {'precision':>10} {'recall':>8} {'IoU':>8} {'img IoU median':>15}")
    for conf, r in report["pixel"].items():
        a = {k: _fmt(v) for k, v in r["all"].items()}
        print(f"{conf:>6} {a['precision']:>10} {a['recall']:>8} {a['iou']:>8} {a['image_iou_median']:>15}")
    print("instance (Ultralytics default conf):", report["instance_default_conf"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
