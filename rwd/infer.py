"""Detection-only inference that logs to CSV (Phase 5).

    python -m rwd.infer --weights runs/<name>/weights/best.pt --source <image folder> --out detections.csv

One CSV row per detected region, plus one row with detection = 0 for an image where nothing was found,
so "looked and found nothing" is also on record. GPS and altitude are read from the file name when it
has the form "..._lat_<deg>_lon_<deg>_alt_<m>m" (the WeedyRice naming); otherwise those cells are empty.

SAFETY: this script only writes a log (and optional overlay images). It must never be connected to a
sprayer, solenoid or any other actuator. Only classes the model was trained on are reported, by name;
nothing is relabelled as "weed".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["timestamp_utc", "image", "detection", "class", "confidence", "x1", "y1", "x2", "y2", "mask_area_px",
          "image_width", "image_height", "lat", "lon", "alt_m", "model", "conf_threshold"]
GPS_RE = re.compile(r"lat_(-?\d+(?:\.\d+)?)_lon_(-?\d+(?:\.\d+)?)(?:_alt_(-?\d+(?:\.\d+)?)m)?")


def parse_gps(name: str) -> dict:
    """{'lat', 'lon', 'alt_m'} from a file name, empty strings when not present."""
    m = GPS_RE.search(name)
    if not m:
        return {"lat": "", "lon": "", "alt_m": ""}
    return {"lat": float(m.group(1)), "lon": float(m.group(2)), "alt_m": float(m.group(3)) if m.group(3) else ""}


def rows_for_image(image: str, width: int, height: int, detections: list[dict], model: str, conf: float,
                   timestamp: str) -> list[dict]:
    """detections: [{'class', 'confidence', 'box': (x1, y1, x2, y2), 'mask_area_px'}]. No detection -> one row."""
    base = {"timestamp_utc": timestamp, "image": image, "image_width": width, "image_height": height,
            "model": model, "conf_threshold": conf, **parse_gps(image)}
    if not detections:
        return [{**dict.fromkeys(FIELDS, ""), **base, "detection": 0}]
    rows = []
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = (round(float(v), 1) for v in d["box"])
        rows.append({**base, "detection": i, "class": d["class"], "confidence": round(float(d["confidence"]), 4),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2, "mask_area_px": d.get("mask_area_px", "")})
    return rows


def write_csv(path: Path, rows: list[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists())
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def detections_from_result(result) -> list[dict]:
    """Ultralytics result -> plain dicts. Mask area is counted at the original image size."""
    import cv2
    import numpy as np

    out = []
    if result.boxes is None:
        return out
    polys = list(result.masks.xy) if result.masks is not None else []
    for i, box in enumerate(result.boxes):
        area = ""
        if i < len(polys) and len(polys[i]) >= 3:
            area = int(round(cv2.contourArea(np.asarray(polys[i], dtype=np.float32))))
        out.append({"class": result.names[int(box.cls)], "confidence": float(box.conf),
                    "box": box.xyxy[0].tolist(), "mask_area_px": area})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, help=".pt, .onnx or an NCNN model folder")
    ap.add_argument("--source", required=True, type=Path, help="image file or folder")
    ap.add_argument("--out", type=Path, default=Path("detections.csv"))
    ap.add_argument("--conf", type=float, default=0.25, help="0.25 is the value the test result was measured at")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280], help="height width, as exported")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--append", action="store_true", help="add to an existing CSV")
    ap.add_argument("--overlays", type=Path, default=None, help="optional folder for images with detections drawn")
    args = ap.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    from rwd.yolo_io import list_images

    images = [args.source] if args.source.is_file() else list_images(args.source)
    images = images[: args.limit] if args.limit else images
    if not images:
        raise SystemExit(f"no images in {args.source}")
    model = YOLO(str(args.weights), task="segment")
    rows, found = [], 0
    for p in images:
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, retina_masks=True)[0]
        dets = detections_from_result(r)
        found += bool(dets)
        h, w = r.orig_shape
        rows += rows_for_image(p.name, w, h, dets, args.weights.name, args.conf,
                               datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if args.overlays:
            args.overlays.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.overlays / f"{p.stem}.jpg"), r.plot())
    write_csv(args.out, rows, args.append)
    print(f"{len(images)} images, {found} with detections, {sum(1 for r in rows if r['detection'])} detections")
    print(f"wrote {args.out}  (log only: detection, no actuator)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
