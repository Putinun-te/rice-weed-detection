# Handoff (2026-11-12, updated on the PC)

First written on the owner's MacBook, updated in the first session on the owner's Windows PC
(WSL2 Ubuntu 26.04, RTX 2080 SUPER 8 GB, driver 620.67).

## Where things stand

| Phase | State |
|---|---|
| 1. Plan | Done and approved. See `docs/PLAN.md`. Track B: public data only, no drone, no Pi. |
| 2. Data tooling | **Done.** 44 synthetic tests pass (Python 3.12). Real dataset built and validated on the PC, see below. Waiting for the owner's OK. |
| 3. Training | Started 2026-11-12 with the owner's OK. `rwd/train.py` + `configs/train.yaml` written; torch 2.15.0+cu132 and ultralytics 8.4.211 in `.venv`; CUDA sees the RTX 2080 SUPER; smoke run (1 epoch, 10% of train) passed. Full run: see below. |
| 4-6 | Not started. |

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
