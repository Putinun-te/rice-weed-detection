"""Export a trained model to ONNX and NCNN for CPU / edge inference (Phase 5).

    python -m rwd.export --weights runs/<name>/weights/best.pt
    python -m rwd.export --weights runs/<name>/weights/best.pt --formats onnx --imgsz 960 1280

The exported files are written next to the weights (inside runs/, which is not in git) and listed in
runs/<name>/export.json. Exporting does not change accuracy claims: only the .pt model was evaluated
on test. Check an exported model on val with rwd.evaluate before relying on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORMATS = ("onnx", "ncnn")


def size_mb(path: Path) -> float:
    """Size of a file, or of every file in a folder (NCNN exports a folder)."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 1024**2, 2)


def export_record(fmt: str, path: Path, imgsz: list[int]) -> dict:
    return {"format": fmt, "path": str(path), "size_mb": size_mb(path), "imgsz_hw": list(imgsz)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[960, 1280], metavar=("H", "W"),
                    help="fixed input size, height width (dataset images are 960x1280)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    from rwd.train import env_info

    weights = args.weights.resolve()
    records = [export_record("pt", weights, args.imgsz)]
    for fmt in args.formats:
        out = Path(YOLO(str(weights)).export(format=fmt, imgsz=args.imgsz, device="cpu"))
        records.append(export_record(fmt, out, args.imgsz))
    report = {"note": "Only the .pt model was evaluated on test. Exports are checked on val only.",
              "source": str(weights), "exports": records, "env": env_info()}
    out_json = weights.parent.parent / "export.json"
    out_json.write_text(json.dumps(report, indent=2))
    for r in records:
        print(f"{r['format']:<5} {r['size_mb']:>7} MB  {r['path']}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
