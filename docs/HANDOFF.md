# Handoff (2026-11-12, updated on the PC)

First written on the owner's MacBook, updated in the first session on the owner's Windows PC
(WSL2 Ubuntu 26.04, RTX 2080 SUPER 8 GB, driver 620.67).

## Where things stand

| Phase | State |
|---|---|
| 1. Plan | Done and approved. See `docs/PLAN.md`. Track B: public data only, no drone, no Pi. |
| 2. Data tooling | **Done.** 44 synthetic tests pass (Python 3.12). Real dataset built and validated on the PC, see below. Waiting for the owner's OK. |
| 3. Training | **Three runs done** (2026-11-12/13): `yolo26n-seg`, `yolo26s-seg`, `yolo26s-seg` + stronger augmentation. On val the small model ties the nano model and the augmentation variant is worse. See below. |
| 4. Test | **Done once for `yolo26n-seg` at conf 0.25** (2026-11-13 00:04). See "Phase 4 result". The other two runs have not touched test. |
| 5-6 | Not started. |

Start with `notebooks/00_overview.ipynb`: it reads the files on disk and shows the state of every
phase, the val comparison, the test result and the recent commits. The owner works from notebooks.

Decisions already made by the owner:
- Model: `yolo26n-seg` (Ultralytics), model name is a config value.
- Dataset: WeedyRice-RGBMS-DB (Mendeley `vt4s83pxx6` v1, CC BY 4.0). Masks only, so segmentation.
  Other candidates were checked and rejected or kept as train-only extras (`DATASET.md`).
- Goal (not a promise): mask mAP50 and recall around 0.80 on the held-out test split.
- Split by capture date/site. Draft: train `2024-06-02`, `2024-06-04`; val `2024-09-30`;
  test `2025-01-15` (Longxuyen, a different site). Change only with the owner's OK.
- Training: on this PC (WSL2, RTX 2080 Super). Studio Lab (T4, 4 h/day, Python 3.9) is the backup.
- Edge speed: measured later on the MacBook M4 CPU as a rough proxy. No Raspberry Pi claims.
- Repo is public on GitHub, AGPL-3.0.

## Phase 2 result (PC, 2026-11-12)

The open issue is resolved. Facts from the real download (`data/inspect.txt` has the raw output):

- `image_metadata.csv` has 3670 rows = 5 per capture (1 RGB + Green, Red, RedEdge, NIR). Config now
  filters with `metadata.row_filter: {sensor_type: RGB}`.
- Files in `RGB/` and `Masks/` use the *standardized* names (`DJI_DateTime_2024_06_02_13_42_0035_lat_..._alt_20.018m.JPG`),
  while the metadata uses `original_filename` (`DJI_20240602134254_0035_D.JPG`). Config now sets
  `metadata.filename_mapping` to `Metadata/filename_mapping.csv` (734 of 734 files mapped).
- Mask = same stem + `.png`, values 0/255 only, 1280x960 like the JPG. No empty mask.
- RGB images per date: 2024-06-02: 115, 2024-06-04: 331, 2024-09-30: 41, 2025-01-15: 247.
- Built `data/weedyrice_seg/` (324 MB, whole images): train 446 / val 41 / test 247 images,
  3439 / 388 / 2516 polygons. Validator OK: no bad lines, no duplicates or near-duplicates across
  splits, no group leakage. Mask vs polygon IoU 0.997 mean (0.954 min) on 120 sampled images.
- Overlays checked by eye for all three splits (`data/overlays/`, 3 per split): polygons sit on the
  lighter weedy-rice patches. The test site (Long Xuyen) has denser and more fragmented patches.
- Risks to keep in mind for Phase 3/4: val has only 41 images; 12 m flights (61 images) exist only
  in test; the split stays as drafted unless the owner changes it.

Environment note: the `.venv` found on the PC was Python 3.14 with no packages, so it was recreated
with `uv python install 3.12` and `PY_BASE="$(uv python find 3.12)" bash scripts/setup_env.sh`.

## Phase 3 first run (PC, 2026-11-12) - val only, not publishable

Run `runs/weedyrice_yolo26n-seg_1280` (yolo26n-seg, imgsz 1280, batch 3, 100 epochs, 49.6 min,
commit 05a5194). Bundle in `bundles/`. Notebook: `notebooks/02_train.ipynb` (`rwd.train --auto`).

- Val (41 images, model selection only): mask precision 0.52, recall 0.49, mAP50 0.47, mAP50-95 0.20.
  Flat over the last 20 epochs (mAP50 0.445 +/- 0.02), so more epochs alone will not help.
- Diagnosis: the instance metric punishes how blobs are split. Masks are amorphous patches cut
  into instances by connected components; the model finds the same area but splits or merges it
  differently. Pixel-level check on val (union of predicted masks vs label): precision 0.80,
  recall 0.75, IoU 0.64 at conf 0.25; precision 0.76, recall 0.86 at conf 0.10.
- Decided by the owner (2026-11-12): pixel-level precision/recall/IoU is the headline metric,
  instance metrics shown next to it (`docs/PLAN.md`, "Metric change"). The test split was used later, see "Phase 4 result".
- Tooling: `rwd/evaluate.py` + `notebooks/03_evaluate.ipynb`. `--split test` needs `--final`, one
  `--conf`, and refuses a second run. Val sweep for the first run (pixel precision / recall / IoU):
  conf 0.05: 0.73 / 0.89 / 0.67; 0.15: 0.78 / 0.84 / 0.68; 0.25: 0.80 / 0.75 / 0.64; 0.5: 0.83 / 0.52 / 0.47.

## Phase 3 comparison (PC, 2026-11-13) - val only, not publishable

Same data, imgsz 1280, 100 epochs, seed 0. Configs: `configs/train.yaml`, `train_s.yaml`,
`train_s_aug.yaml` (scale 0.7, hsv_v 0.5, hsv_s 0.8). The s runs use batch 2 (batch 3 filled 8 GB).
Val is 41 images from one date, so small differences mean nothing.

| Run | Minutes | Best pixel IoU (conf) | Pixel P / R at that conf | Instance mask P / R / mAP50 |
|---|---|---|---|---|
| `yolo26n-seg` | 50 | 0.678 (0.15) | 0.78 / 0.84 | 0.52 / 0.49 / 0.47 |
| `yolo26s-seg` | 79 | 0.686 (0.10) | 0.81 / 0.82 | 0.60 / 0.50 / 0.52 |
| `yolo26s-seg` + aug | 100 | 0.664 (0.10) | 0.82 / 0.78 | 0.51 / 0.44 / 0.41 |

Reading: a bigger model and stronger augmentation do not move val. The limit looks like the data
(two training dates), not model size. `yolo26n-seg` stays the model.

## Phase 4 result (PC, 2026-11-13) - held-out test, publishable with recall

`yolo26n-seg` at conf 0.25, evaluated once on test (247 images, Long Xuyen, 2025-01-15, a site and
date not in train or val). File: `runs/weedyrice_yolo26n-seg_1280/eval_test.json`, written
2026-11-13 00:04 (shown in the test cell of `notebooks/03_evaluate.ipynb`). Conf 0.25 was the notebook
default, not the val optimum (0.15), and it was not tuned on test.

| Group | Images | Pixel precision | Pixel recall | Pixel IoU |
|---|---|---|---|---|
| All | 247 | 0.889 | 0.697 | 0.641 |
| 12 m | 61 | 0.960 | 0.665 | 0.647 |
| 20 m | 186 | 0.859 | 0.712 | 0.638 |

Instance masks on test: precision 0.283, recall 0.223, mAP50 0.181, mAP50-95 0.060.
The goal of mask mAP50 and recall around 0.80 is not met. At pixel level the model is rarely wrong
when it marks weedy rice but misses about 30% of the weedy-rice area.

Rules from here: the script refuses a second test run for this model. Any other model evaluated on
this test split is a second look and must be labelled as such. Do not re-run test at another conf.

## Next steps

1. Owner decides: stop model work here and go to Phase 5 with `yolo26n-seg`, or try one more idea
   on val (a semantic-segmentation model, or CoFly-WeedDB as train-only extra, see `DATASET.md`).
2. No drone is available, so new field data is not planned. The only UAV weedy-rice dataset found
   (2026-11-13 search) is the one already used.
3. Notebooks keep their outputs locally; the `nbstrip` git filter (set by `scripts/setup_env.sh`)
   commits them without outputs, so no third-party image reaches git.

## PC setup (WSL2 Ubuntu, done by the next session with the owner)

1. Check the GPU from WSL: `nvidia-smi` must show the RTX 2080 SUPER. The NVIDIA driver is
   installed on Windows only, never inside Ubuntu.
2. `sudo apt update && sudo apt install -y git python3-venv python3-pip` (needs the owner's password).
3. `cd ~ && git clone https://github.com/Putinun-te/rice-weed-detection.git`
4. `bash ~/rice-weed-detection/scripts/setup_env.sh` (creates `.venv` and kernel `Python (rwd)`).
5. `.venv/bin/python -m pytest -q` must pass.
6. Data: `bash scripts/studiolab_data.sh` works on WSL too (needs about 7 GB free during download). Done on the PC; took about 10 minutes.
7. Phase 3 only, after the owner's OK: install PyTorch with CUDA and Ultralytics into `.venv`,
   then confirm `torch.cuda.is_available()` and the device name before any training.

## Notes for Phase 3 (training)

- One config and one script for both PC and Studio Lab. Resumable (`save_period`, `resume=True`),
  fixed seed, record GPU name, VRAM, versions and commit hash in `metrics.json`.
- 8 GB VRAM: start with imgsz 1280 and an automatic or small batch; fall back to tiles (640)
  if memory runs out. AMP works on Turing.
- At the end of every run, bundle `best.pt`, `last.pt`, `metrics.json` and plots so nothing is lost.
- WSL may be killed for memory: if so, raise the limit in `%UserProfile%\.wslconfig`.

## Files to know

- `rwd/validate.py` dataset checks (classes, empty labels, duplicates, group leakage)
- `rwd/prepare_weedyrice.py` raw WeedyRice to YOLO-seg (date split, tiling, mask to polygons)
- `rwd/download.py` Mendeley/Roboflow/Kaggle downloads (keys from env only), nested zip support
- `rwd/viz.py` label overlays; `notebooks/01_data_setup.ipynb` the same steps as a notebook
- `scripts/setup_env.sh`, `scripts/studiolab_data.sh`
