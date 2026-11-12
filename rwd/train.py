"""Train a YOLO model on the prepared dataset (Phase 3). One script for the PC and Studio Lab.

- The model name is a config value (configs/train.yaml).
- Resumable: `--resume` continues from runs/<name>/weights/last.pt.
- After training it evaluates on the *val* split only. The test split is never touched here:
  it is used once, in Phase 4. Val numbers are for model selection, not for publishing.
- Writes runs/<name>/metrics.json (GPU, VRAM, versions, commit hash, val metrics) and a bundle
  with best.pt, last.pt, metrics.json and plots in bundles/.

Usage:
    python -m rwd.train --config configs/train.yaml --check     # GPU and dataset check only
    python -m rwd.train --config configs/train.yaml --smoke     # 1 short epoch on 10% of train
    python -m rwd.train --config configs/train.yaml             # full run
    python -m rwd.train --config configs/train.yaml --resume    # continue a stopped run
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REQUIRED = ("model", "data", "project", "name", "imgsz", "epochs", "batch", "seed")
BUNDLE_FILES = ("weights/best.pt", "weights/last.pt", "metrics.json", "args.yaml", "results.csv")
RESERVED_EXTRA = {"data", "model", "project", "name", "resume", "exist_ok", "split"}


def load_train_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    missing = [k for k in REQUIRED if cfg.get(k) is None]
    if missing:
        raise ValueError(f"{path.name} is missing: {missing}")
    for key in ("data", "project"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (path.parent / p).resolve()
    extra = cfg.get("extra") or {}
    bad = RESERVED_EXTRA & set(extra)
    if bad:
        raise ValueError(f"extra must not set {sorted(bad)}; use the top-level keys")
    cfg["extra"] = extra
    return cfg


def task_of(model_name: str) -> str:
    """'segment' for *-seg models, else 'detect'. Other tasks are not used in this project."""
    return "segment" if Path(model_name).stem.endswith("-seg") else "detect"


def train_args(cfg: dict, smoke: bool = False) -> dict:
    """Arguments for YOLO.train(). Smoke = 1 epoch on 10% of train, in its own run folder."""
    args = {
        "data": str(cfg["data"]), "project": str(cfg["project"]), "name": cfg["name"],
        "imgsz": cfg["imgsz"], "epochs": cfg["epochs"], "batch": cfg["batch"], "seed": cfg["seed"],
        "patience": cfg.get("patience", 30), "workers": cfg.get("workers", 4),
        "deterministic": cfg.get("deterministic", True), "amp": cfg.get("amp", True),
        "save_period": cfg.get("save_period", 5), "device": cfg.get("device", 0),
        "exist_ok": False, **cfg["extra"],
    }
    if smoke:
        args.update(name=f"{cfg['name']}_smoke", epochs=1, fraction=0.1, exist_ok=True, save_period=-1, plots=False)
    return args


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def env_info(device=0) -> dict:
    import torch
    import ultralytics

    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": None, "vram_gb": None, **git_info(),
    }
    if torch.cuda.is_available() and str(device) != "cpu":
        idx = int(str(device).split(",")[0])
        info["gpu"] = torch.cuda.get_device_name(idx)
        info["vram_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1024**3, 2)
    return info


def check_device(cfg: dict, env: dict) -> None:
    if str(cfg.get("device", 0)) != "cpu" and not env["cuda_available"]:
        raise SystemExit("CUDA is not available but the config asks for a GPU. Check nvidia-smi, or set device: cpu.")


def write_metrics(run_dir: Path, cfg: dict, env: dict, val_metrics: dict | None, extra: dict | None = None) -> Path:
    data = {
        "note": "val split only, for model selection. Not a publishable result; the test split is used in Phase 4.",
        "model": cfg["model"], "task": task_of(cfg["model"]), "data": str(cfg["data"]),
        "imgsz": cfg["imgsz"], "epochs": cfg["epochs"], "batch": cfg["batch"], "seed": cfg["seed"],
        "env": env, "val": val_metrics, **(extra or {}),
    }
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    return out


def bundle(run_dir: Path, bundles_dir: Path) -> Path:
    """tar.gz with the weights, metrics.json, args, results.csv and plots of one run."""
    bundles_dir.mkdir(parents=True, exist_ok=True)
    out = bundles_dir / f"{run_dir.name}.tar.gz"
    files = [run_dir / f for f in BUNDLE_FILES if (run_dir / f).exists()]
    files += sorted(p for ext in ("*.png", "*.jpg") for p in run_dir.glob(ext))
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"{run_dir.name}/{f.relative_to(run_dir)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true", help="print GPU and versions, check the dataset, then stop")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of train to test the pipeline and memory")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/weights/last.pt")
    args = ap.parse_args(argv)

    cfg = load_train_config(args.config)
    if not cfg["data"].exists():
        raise SystemExit(f"{cfg['data']} not found. Build the dataset first (rwd.prepare_weedyrice).")
    env = env_info(cfg.get("device", 0))
    print(json.dumps(env, indent=2))
    check_device(cfg, env)
    if args.check:
        return 0

    from ultralytics import YOLO

    targs = train_args(cfg, smoke=args.smoke)
    run_dir = Path(targs["project"]) / targs["name"]
    last = run_dir / "weights" / "last.pt"
    start = time.time()
    if args.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        if run_dir.exists() and not args.smoke:
            raise SystemExit(f"{run_dir} already exists. Use --resume, or change `name` in the config.")
        model = YOLO(cfg["model"])
        model.train(**targs)
    train_minutes = round((time.time() - start) / 60, 1)

    best = run_dir / "weights" / "best.pt"
    val = YOLO(str(best)).val(data=str(cfg["data"]), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"],
                              device=cfg.get("device", 0), plots=False, project=str(run_dir), name="val_check", exist_ok=True)
    val_metrics = {k: round(float(v), 4) for k, v in val.results_dict.items()}
    out = write_metrics(run_dir, cfg, env, val_metrics, {"smoke": args.smoke, "train_minutes_this_session": train_minutes})
    print(f"wrote {out}")
    print(json.dumps(val_metrics, indent=2))
    if not args.smoke:
        print(f"bundle: {bundle(run_dir, REPO / 'bundles')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
