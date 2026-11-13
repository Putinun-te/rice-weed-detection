"""Collect the validation results of every trained run into docs/results/ (JSON + three charts).

    .venv/bin/python scripts/make_model_report.py

Reads runs/<name>/{eval_val.json,metrics.json,results.csv} (and eval_test.json for the chosen run).
Validation numbers are for comparing models only; they were used to choose the model. The charts
contain numbers only, no dataset images.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/results"

# name, label, marker, categorical slot (hex from the validated reference palette, light surface)
RUNS = [
    ("weedyrice_yolo26n-seg_1280", "yolo26n-seg (chosen)", "o", "#2a78d6"),
    ("weedyrice_yolo26s-seg_1280", "yolo26s-seg", "s", "#eb6834"),
    ("weedyrice_yolo26s-seg_1280_aug", "yolo26s-seg + strong aug", "^", "#1baf7a"),
    ("weedyrice_cofly_yolo26n-seg_1280", "yolo26n-seg + CoFly", "D", "#eda100"),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load(run: str) -> dict:
    d = ROOT / "runs" / run
    val, met = json.loads((d / "eval_val.json").read_text()), json.loads((d / "metrics.json").read_text())
    epochs = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(d / "results.csv"))]
    pixel = {c: {k: v["all"][k] for k in ("precision", "recall", "iou", "image_iou_median")} for c, v in val["pixel"].items()}
    return {"model": met["model"], "batch": met["batch"], "epochs_run": len(epochs),
            "train_minutes": met["train_minutes_this_session"], "pixel_val": pixel,
            "instance_val": {k: v for k, v in val["instance_default_conf"].items()}, "curve": epochs}


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def legend(fig, ax) -> None:
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {label: load(run) for run, label, _, _ in RUNS}
    summary = {"note": "Validation split (41 images, 2024-09-30), used to choose the model. Not a held-out result. "
                       "Pixel scores are for class weedy_rice only. Test was evaluated once, for the chosen model only.",
              "runs": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in data.items()}}
    (OUT / "val_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    # 1. validation: IoU vs conf, and precision vs recall
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for run, label, mk, col in RUNS:
        px = data[label]["pixel_val"]
        confs = list(px)
        a1.plot([float(c) for c in confs], [px[c]["iou"] for c in confs], marker=mk, color=col, lw=2, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        a2.plot([px[c]["recall"] for c in confs], [px[c]["precision"] for c in confs], marker=mk, color=col, lw=2,
                ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    style(a1, "confidence threshold", "pixel IoU (weedy_rice)", "Validation: IoU by confidence")
    style(a2, "pixel recall", "pixel precision", "Validation: precision vs recall (points = confidence 0.05 ... 0.5)")
    a1.set_ylim(0.2, 0.75); a2.set_ylim(0.65, 0.95); a2.set_xlim(0.2, 0.95)
    legend(fig, a1)
    fig.suptitle("Validation split only (41 images): used to choose the model, not a held-out result",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "val_comparison.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 2. training curves on val
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, key, name in ((axes[0], "metrics/recall(M)", "instance mask recall"), (axes[1], "metrics/mAP50(M)", "instance mask mAP50")):
        for run, label, mk, col in RUNS:
            ep = data[label]["curve"]
            ax.plot([r["epoch"] for r in ep], [r[key] for r in ep], color=col, lw=1.6, label=label)
        style(ax, "epoch", name, f"Validation {name} during training")
        ax.set_ylim(0, 0.8)
    legend(fig, axes[0])
    fig.suptitle("Validation split only (41 images). Instance metrics are low for all runs: patches have no natural instance boundaries",
                 x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "training_curves_val.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 3. test result of the chosen model (the only publishable one), by flight height
    t = json.loads((ROOT / "runs/weedyrice_yolo26n-seg_1280/eval_test.json").read_text())["pixel"]["0.25"]
    groups = [("All (247)", t["all"]), ("12 m (61)", t["by_altitude"]["12m"]), ("20 m (186)", t["by_altitude"]["20m"])]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor=SURFACE)
    width = 0.26
    for i, (key, col) in enumerate((("precision", "#2a78d6"), ("recall", "#eb6834"), ("iou", "#1baf7a"))):
        xs = [j + (i - 1) * (width + 0.02) for j in range(3)]
        vals = [g[1][key] for g in groups]
        bars = ax.bar(xs, vals, width, color=col, label="IoU" if key == "iou" else key, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 1.05); ax.axhline(0.8, color="#c9c8c3", lw=1, ls="--")
    ax.text(2.45, 0.815, "goal: recall about 0.80", ha="right", fontsize=8, color=MUTED)
    style(ax, "test images by flight height", "pixel score (weedy_rice)", "Test result, yolo26n-seg, confidence 0.25, evaluated once")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5, frameon=False, labelcolor=INK, bbox_to_anchor=(0, 0.98))
    ax.set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(OUT / "test_result.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    print("wrote", *(p.name for p in sorted(OUT.glob("*"))), sep="\n  ")
    for label, d in data.items():
        px = d["pixel_val"]
        best = max(px.items(), key=lambda kv: kv[1]["iou"])
        i = d["instance_val"]
        print(f"{label:<26} {d['model']:<16} batch {d['batch']} {d['train_minutes']:>6.1f} min | best IoU {best[1]['iou']:.3f} @ {best[0]} "
              f"(P {best[1]['precision']:.3f} R {best[1]['recall']:.3f}) | 0.25: P {px['0.25']['precision']:.3f} R {px['0.25']['recall']:.3f} "
              f"IoU {px['0.25']['iou']:.3f} | inst P {i['metrics/precision(M)']:.3f} R {i['metrics/recall(M)']:.3f} "
              f"mAP50 {i['metrics/mAP50(M)']:.3f} mAP50-95 {i['metrics/mAP50-95(M)']:.3f}")


if __name__ == "__main__":
    main()
