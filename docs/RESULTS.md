# Results and limits

Final summary of the project as of 2026-11-13. Track B: public data only, no drone, no edge board.
Detection only: the model output is a log, never a command to a sprayer or any actuator.

## What was built

A reproducible pipeline for finding weedy rice in UAV RGB images of rice fields: dataset preparation
and validation, training, pixel-level evaluation with a guarded test split, ONNX/NCNN export, and
inference that logs to CSV. 70 tests on synthetic data. One notebook per step in `notebooks/`.

## Data and model

- Dataset: WeedyRice-RGBMS-DB (Nguyen et al., Mendeley `vt4s83pxx6` v1, CC BY 4.0), RGB images only.
  Third-party data, not the project owner's. Details and attribution: `DATASET.md`.
- One class: `weedy_rice` (the dataset labels weedy rice against background, nothing else).
- Split by capture date and site, never by random image:
  train 446 images (2024-06-02, 2024-06-04), val 41 (2024-09-30), test 247 (2025-01-15, Long Xuyen).
- Model: Ultralytics `yolo26n-seg`, input 1280, 100 epochs, RTX 2080 SUPER. Config: `configs/train.yaml`.

## Result on the held-out test split

Evaluated once, at confidence 0.25. The test images are from a site and date that are in neither
train nor val. Headline metric: pixel-level precision, recall and IoU of the union of predicted
masks against the label mask (reason: `docs/PLAN.md`, "Metric change").

| Group | Images | Pixel precision | Pixel recall | Pixel IoU |
|---|---|---|---|---|
| All test images | 247 | 0.889 | 0.697 | 0.641 |
| Flight height 12 m | 61 | 0.960 | 0.665 | 0.647 |
| Flight height 20 m | 186 | 0.859 | 0.712 | 0.638 |

Instance-level mask metrics on the same split: precision 0.283, recall 0.223, mAP50 0.181,
mAP50-95 0.060.

Reading:
- When the model marks an area as weedy rice it is usually right, but it misses about 30% of the
  weedy-rice area.
- The original goal (mask mAP50 and recall around 0.80) is **not met**.
- Instance metrics are low because weedy-rice patches have no natural instance boundaries; the model
  finds much of the same area but splits it differently from the labels.
- Confidence 0.25 was the default, not the value that looked best on validation, and it was not
  tuned on test. The test split was not re-run at another confidence.

Three other variants (a larger model, stronger augmentation, and CoFly-WeedDB added as a second
class) were compared on the validation split only and none was clearly better, so they were not
evaluated on test. They are kept as optional configs. Details: `docs/HANDOFF.md`.

## Export and speed

ONNX and NCNN exports (fixed input 960x1280, about 11 MB each) give the same pixel scores as the
PyTorch model on the validation split. Only the PyTorch model was evaluated on test.

CPU speed, whole predict call per 960x1280 image, 20 images:

| Computer | PyTorch | ONNX Runtime | NCNN | Peak RAM |
|---|---|---|---|---|
| Desktop, Intel Core i9-9900KS (16 logical cores) | 134 ms | 153 ms | 213 ms | 1.1 to 1.4 GB |
| MacBook Pro, Apple M4 (10 logical cores) | 126 ms | 107 ms | 70 ms | 0.8 to 1.1 GB |

Times are medians. The M4 run (2026-11-13, `scripts/mac_benchmark.sh`) had other apps open.
These are laptop/desktop CPU numbers, a rough proxy only.

## Data credit

Trained and evaluated on **WeedyRice-RGBMS-DB** (CC BY 4.0), a third-party dataset that is not part of this
repository: Nguyen, V. H., Le, C.-D., Truong, M.-T., Bui, M.-P., Le, T.-P. (2025). *WeedyRice-RGBMS-DB: A
dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice.* Mendeley Data,
V1, doi:[10.17632/vt4s83pxx6.1](https://doi.org/10.17632/vt4s83pxx6.1); Data in Brief,
doi:[10.1016/j.dib.2025.112237](https://doi.org/10.1016/j.dib.2025.112237).
Source files for the numbers on this page: [`results/`](results/).
Model files: [release v1.0.0](https://github.com/Putinun-te/rice-weed-detection/releases/tag/v1.0.0).

## What cannot be claimed

- Field validation in Thai rice fields, or on any field, season, camera or altitude outside the
  test split above.
- Any result on a drone, on a Raspberry Pi or on another edge board. `docs/EDGE_CHECKLIST.md` is a
  checklist that has not been run on hardware.
- Detection of weeds other than weedy rice.
- Fitness for driving a sprayer. Detection only.

Known weak points: one dataset from one region; training data from two capture dates; a validation
split of 41 images from one date; 12 m flights appear only in the test split.

## Reproduce

```bash
bash scripts/setup_env.sh                                  # project venv + Jupyter kernel "Python (rwd)"
# then, in order, with the "Python (rwd)" kernel:
#   notebooks/01_data_setup.ipynb    download and build the dataset
#   notebooks/02_train.ipynb         train (GPU), resumable
#   notebooks/03_evaluate.ipynb      validation sweep; the test cell is guarded and runs once
#   notebooks/05_export_infer.ipynb  export, CPU benchmark, CSV inference
#   notebooks/00_overview.ipynb      status of everything, read-only
.venv/bin/python -m pytest -q                              # tests, synthetic data only
```

Training is seeded (`seed: 0`, deterministic), but exact numbers can still differ slightly across
GPUs and library versions. The versions used are recorded in `runs/<name>/metrics.json`.
