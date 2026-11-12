"""Tests for the training wrapper that need neither torch nor a GPU."""

from __future__ import annotations

import json
import tarfile

import pytest
import yaml

from rwd.train import run_state, bundle, check_device, load_train_config, task_of, train_args, write_metrics

BASE = {"model": "yolo26n-seg.pt", "data": "ds/data.yaml", "project": "runs", "name": "run1",
        "imgsz": 1280, "epochs": 100, "batch": 4, "seed": 0}


def write(tmp_path, **over):
    cfg = {**BASE, **over}
    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_paths_resolve_against_config_folder(tmp_path):
    cfg = load_train_config(write(tmp_path))
    assert cfg["data"] == (tmp_path / "ds/data.yaml").resolve()
    assert cfg["project"] == (tmp_path / "runs").resolve()


def test_missing_key_and_reserved_extra_fail(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_train_config(write(tmp_path, seed=None))
    with pytest.raises(ValueError, match="extra must not set"):
        load_train_config(write(tmp_path, extra={"split": "test"}))


def test_task_follows_model_name():
    assert task_of("yolo26n-seg.pt") == "segment"
    assert task_of("yolo26n.pt") == "detect"


def test_train_args_full_and_smoke(tmp_path):
    cfg = load_train_config(write(tmp_path, extra={"flipud": 0.5}))
    full = train_args(cfg)
    assert full["epochs"] == 100 and full["flipud"] == 0.5 and full["exist_ok"] is False
    assert "fraction" not in full and "split" not in full
    smoke = train_args(cfg, smoke=True)
    assert smoke["epochs"] == 1 and smoke["fraction"] == 0.1 and smoke["name"] == "run1_smoke"


def test_gpu_required_unless_cpu(tmp_path):
    cfg = load_train_config(write(tmp_path))
    with pytest.raises(SystemExit):
        check_device(cfg, {"cuda_available": False})
    check_device({**cfg, "device": "cpu"}, {"cuda_available": False})


def test_metrics_and_bundle(tmp_path):
    cfg = load_train_config(write(tmp_path))
    run = tmp_path / "runs/run1"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best.pt").write_bytes(b"w")
    (run / "weights/last.pt").write_bytes(b"w")
    (run / "results.png").write_bytes(b"p")
    env = {"gpu": "Fake GPU", "vram_gb": 8.0, "commit": "abc", "cuda_available": True}
    out = write_metrics(run, cfg, env, {"metrics/recall(M)": 0.5})
    data = json.loads(out.read_text())
    assert data["env"]["gpu"] == "Fake GPU" and data["task"] == "segment" and "val split only" in data["note"]
    tar_path = bundle(run, tmp_path / "bundles")
    names = set(tarfile.open(tar_path).getnames())
    assert {"run1/weights/best.pt", "run1/weights/last.pt", "run1/metrics.json", "run1/results.png"} <= names


def test_run_state(tmp_path):
    run = tmp_path / "run1"
    assert run_state(run) == "new"
    (run / "weights").mkdir(parents=True)
    assert run_state(run) == "new"  # killed before the first checkpoint
    (run / "weights/last.pt").write_bytes(b"w")
    assert run_state(run) == "resumable"
    (run / "metrics.json").write_text("{}")
    assert run_state(run) == "finished"
