"""CPU latency and memory of a model file, one format per process (Phase 5).

    python -m rwd.benchmark --weights best.pt best.onnx best_ncnn_model --images data/weedyrice_seg/images/val

Numbers describe THIS computer's CPU only. They are a rough proxy and say nothing about a Raspberry Pi
or any other board. Speed only: accuracy is measured by rwd.evaluate.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def summarize(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    return {"images": len(s), "median_ms": round(statistics.median(s), 1), "mean_ms": round(statistics.fmean(s), 1),
            "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1), "fps_median": round(1000 / statistics.median(s), 2)}


def run_one(weights: Path, images: list[Path], imgsz: list[int], conf: float, warmup: int) -> dict:
    import resource

    from ultralytics import YOLO

    model = YOLO(str(weights), task="segment")
    kw = dict(imgsz=imgsz, conf=conf, device="cpu", verbose=False, retina_masks=True)
    for p in images[:warmup]:
        model.predict(str(p), **kw)
    times = []
    for p in images:
        t = time.perf_counter()
        model.predict(str(p), **kw)
        times.append((time.perf_counter() - t) * 1000)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: kB
    return {"weights": weights.name, **summarize(times), "peak_rss_mb": round(peak_mb)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path, nargs="+")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)  # internal: one model, print JSON
    args = ap.parse_args(argv)

    from rwd.yolo_io import list_images

    images = list_images(args.images)[: args.n]
    if args.single:
        print("RESULT " + json.dumps(run_one(args.weights[0], images, args.imgsz, args.conf, args.warmup)))
        return 0

    results = []
    for w in args.weights:  # one process each, so peak memory is not mixed between formats
        cmd = [sys.executable, "-m", "rwd.benchmark", "--single", "--weights", str(w), "--images", str(args.images),
               "--n", str(args.n), "--conf", str(args.conf), "--warmup", str(args.warmup), "--imgsz", *map(str, args.imgsz)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(f"benchmark failed for {w}:\n{out.stderr[-2000:]}")
        results.append(json.loads(line[7:]))
    import os

    report = {"note": "CPU of this computer only. Rough proxy, not a Raspberry Pi or drone result. Speed, not accuracy.",
              "cpu": cpu_name(), "logical_cores": os.cpu_count(), "platform": platform.platform(),
              "imgsz_hw": args.imgsz, "conf": args.conf, "results": results}
    print(f"CPU: {report['cpu']} ({report['logical_cores']} logical cores)")
    print(f"{'model':<22} {'median ms':>10} {'p90 ms':>8} {'fps':>6} {'peak RSS MB':>12}")
    for r in results:
        print(f"{r['weights']:<22} {r['median_ms']:>10} {r['p90_ms']:>8} {r['fps_median']:>6} {r['peak_rss_mb']:>12}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
