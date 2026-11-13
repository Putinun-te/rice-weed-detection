"""Pixel-level evaluation helpers. Synthetic arrays only; no model, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from rwd.evaluate import altitude_band, error_overlay, label_mask, main, pixel_counts, rasterize, scores, summarize

SQ = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], float)


def test_rasterize_union_and_degenerate():
    m = rasterize([SQ, SQ + 1, np.array([[0, 0], [1, 1]])], 10, 10)
    assert m.dtype == bool and m[4, 4] and m[7, 7] and not m[0, 0]


def test_label_mask_reads_yolo_label(clean_dataset):
    m = label_mask(clean_dataset / "images/train/a.png", 48, 64)  # square x,y in 0.1..0.3
    assert m[int(48 * 0.2), int(64 * 0.2)] and not m[40, 60]
    (clean_dataset / "labels/train/a.txt").unlink()
    assert not label_mask(clean_dataset / "images/train/a.png", 48, 64).any()


def test_counts_and_scores():
    gt = np.zeros((4, 4), bool); gt[:2] = True          # 8 px
    pred = np.zeros((4, 4), bool); pred[1:3] = True      # 8 px, 4 overlap
    c = pixel_counts(pred, gt)
    assert c == {"tp": 4, "fp": 4, "fn": 4}
    s = scores(**c)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["iou"] == pytest.approx(1 / 3)
    assert scores(0, 0, 0) == {"precision": None, "recall": None, "iou": None}


def test_summarize_pools_pixels_and_groups():
    rows = [{"tp": 90, "fp": 10, "fn": 0, "altitude": "20m"}, {"tp": 0, "fp": 0, "fn": 100, "altitude": "12m"}]
    s = summarize(rows)
    assert s["images"] == 2 and s["precision"] == 0.9 and s["recall"] == pytest.approx(0.4737, abs=1e-4)
    assert s["image_iou_min"] == 0.0
    by = summarize(rows, "altitude")
    assert set(by) == {"12m", "20m"} and by["12m"]["recall"] == 0.0 and by["20m"]["recall"] == 1.0


def test_altitude_band():
    assert altitude_band("20.018m") == "20m" and altitude_band("12.3m") == "12m" and altitude_band("21.0m") == "20m"
    assert altitude_band("") == "unknown" and altitude_band(None) == "unknown"


def test_error_overlay_colours():
    img = np.zeros((2, 3, 3), np.uint8)
    pred = np.array([[1, 1, 0], [0, 0, 0]], bool)
    gt = np.array([[1, 0, 1], [0, 0, 0]], bool)
    out = error_overlay(img, pred, gt, alpha=1.0)
    assert tuple(out[0, 0]) == (0, 200, 0) and tuple(out[0, 1]) == (0, 80, 255) and tuple(out[0, 2]) == (255, 0, 0)
    assert tuple(out[1, 1]) == (0, 0, 0)


def test_test_split_is_guarded(tmp_path):
    w = tmp_path / "run/weights/best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"w")
    with pytest.raises(SystemExit, match="--final"):
        main(["--weights", str(w), "--split", "test"])
    with pytest.raises(SystemExit, match="exactly one --conf"):
        main(["--weights", str(w), "--split", "test", "--final", "--conf", "0.1", "0.25"])
    (tmp_path / "run/eval_test.json").write_text("{}")
    with pytest.raises(SystemExit, match="already used"):
        main(["--weights", str(w), "--split", "test", "--final", "--conf", "0.25"])


def test_tag_is_refused_on_test_split(tmp_path):
    with pytest.raises(SystemExit, match="val only"):
        main(["--weights", str(tmp_path / "w/weights/best.onnx"), "--split", "test", "--final", "--tag", "onnx"])


def test_summary_prints_na_when_nothing_predicted(monkeypatch, tmp_path, capsys):
    """Found on the Mac smoke run: precision is None when no pixel is predicted, and the table crashed."""
    import rwd.evaluate as ev

    undefined = {"images": 4, "precision": None, "recall": 0.0, "iou": 0.0, "image_iou_median": 0.0, "image_iou_min": 0.0}
    report = {"images": 4, "pixel": {"0.5": {"all": undefined}}, "instance_default_conf": {}}
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: report)
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    assert main(["--weights", str(weights), "--split", "val", "--conf", "0.5", "--device", "cpu"]) == 0
    assert "n/a" in capsys.readouterr().out
    assert (tmp_path / "run" / "eval_val.json").exists()
