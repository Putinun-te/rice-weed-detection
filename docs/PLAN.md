# Phase 1 Plan (draft, awaiting review)

Status: draft. Nothing in this file is a result. No model has been trained yet.

## Situation

- No drone access and no Raspberry Pi 5 right now. Only a MacBook and Amazon SageMaker Studio Lab.
- Therefore the project runs as **Track B: offline-first, public datasets only**.
- The model and numbers from the earlier course project are not reused.
- The Rumex obtusifolius grassland dataset is not rice data and is not used.

## Goal

Build a reproducible pipeline for weed detection in rice fields from UAV RGB images:
validate data, train on Studio Lab, evaluate on a held-out split, export to NCNN/ONNX,
and run detection-only inference that logs to CSV.

## What can and cannot be claimed (Track B)

Can claim:
- Metrics on the held-out split of the public dataset used, with recall always reported.

Cannot claim:
- Field validation in Thai rice fields.
- Results on a drone, or any on-board (on-drone) results.
- Raspberry Pi 5 or other edge-board performance. Laptop CPU numbers are a rough proxy only.
- Generalisation to fields, seasons, cameras or altitudes not in the held-out split.
- Fitness for driving a sprayer. The model is detection only and must never be connected to an actuator.

## Model

- Main model: `yolo26n-seg` (Ultralytics), because the primary dataset (WeedyRice-RGBMS-DB) has masks, not boxes.
  Model name is a config value.
- Optional comparison later: `yolo11n-seg`, `yolo12n-seg`, same split, seed, imgsz and epochs.
- Output at inference stays detection only (weed regions logged to CSV, no actuator).

## Target (a goal, not a result)

- Goal: about 0.80 or higher on the held-out **test** split (different site and date from training),
  measured as mask mAP50 and mask recall for `weedy_rice`, reported together with precision.
- Pixel accuracy is not used as the headline number: background dominates the image, so it would look high
  even for a model that finds no weeds.
- Not guaranteed. The dataset is small (734 images) and the test site differs from the training site.
  If the goal is missed, the report says so and shows where the model fails.

## Classes

- Start: depends on the dataset. The best candidate (WeedyRice-RGBMS-DB) only has weedy rice vs background.
- Target later: `rice`, `weed`, then weed types (grass, sedge, broadleaf, weedy rice).
- Early-stage grass weeds and weedy rice look like rice. Expect low recall there.

## Images and GSD

- GSD = (altitude x sensor width) / (focal length x image width in px).
- Small weeds need roughly 0.3-0.5 cm/px or finer.
- Train on tiles (1280 px, about 20% overlap) or imgsz 1280.
- Tiling happens only after the split, so tiles from one image never cross splits.

## Split policy

- Split by field / date / season, never by random image.
- If a dataset has no field/date metadata, the split is approximate and every number from it is labelled as such.
- The test split is used once, at the end.

## Training

- Main: the owner's PC, WSL2 Ubuntu with an RTX 2080 Super (8 GB, Turing). No session limit.
- Backup: SageMaker Studio Lab (T4, 4 h per session and per 24 h, Python 3.9, about 15 GB disk).
- One config and one script for both. Check the GPU (`nvidia-smi`, `torch.cuda.is_available()`) first
  and record GPU, versions and commit hash in metrics.json.
- Resumable runs: small `save_period`, `resume=True`, fixed seed.
- At the end of every run, bundle best.pt, last.pt, metrics.json and plots.
- The repo is public, so no token is needed to clone. Never put tokens in notebook cells.

## Edge readiness (no hardware)

- Export NCNN and ONNX.
- Benchmark latency and memory on the MacBook CPU (state the chip model) as a rough proxy.
- Target board is a config value. A Pi 5 checklist is written but marked untested on hardware.
- Inference script logs detections with timestamp (and GPS if available) to CSV. Detection only.

## Risks

- Leakage between splits from overlapping or near-duplicate images.
- Domain gap: public data is not from Thai fields, and may differ in camera, altitude and season.
- Segmentation masks converted to boxes can give large, misleading boxes for scattered weeds.
- Roboflow datasets are often pre-augmented and resized, which creates near-duplicates.
- Licenses that forbid redistribution or commercial use.
- T4-class GPU and 4 h sessions limit imgsz and epochs.

## Lessons from the old course repos (read-only review)

Prac.IoT-Weedetection:
- COCO export had two categories, but every annotation (4199 train, 662 test) used category 1
  (Rumex, "ridderzuring"). Category 0 "grass-weeds" was a Roboflow supercategory with zero instances.
  The model effectively learned one class, while the yaml declared two.
- The converter copied COCO `category_id` straight into YOLO class ids. New tooling must map
  categories explicitly and fail on unknown or empty classes.
- The split was by image (Roboflow train/test), with no field or date information.
- README speed figures for Raspberry Pi had no recorded benchmark behind them. Not reused.

Practical-IoT-Drone-Cost-Optimizing:
- The `yolo26n_ncnn_model` there is the stock COCO model (80 classes, metadata `coco.yaml`),
  not a weed model. Its `model_ncnn.py` is only a random-tensor smoke test.
- Useful as a pattern: Ultralytics NCNN export layout (`model.ncnn.param`, `model.ncnn.bin`, `metadata.yaml`).
- Weights (`*.pt`) were committed to git there. This repo ignores them.

## Open questions

1. Confirm licenses and metadata of candidate datasets (pending read-only check of source pages).
2. Code transfer to Studio Lab: zip upload or private GitHub repo.
3. Detection (boxes) or segmentation (YOLO-seg) if the main dataset only has masks.
