# Handoff (2026-11-12)

Written at the end of the planning/data-tooling session on the owner's MacBook, for the next
Claude Code session on the owner's Windows PC (WSL2, RTX 2080 Super 8 GB).

## Where things stand

| Phase | State |
|---|---|
| 1. Plan | Done and approved. See `docs/PLAN.md`. Track B: public data only, no drone, no Pi. |
| 2. Data tooling | Code done and tested (38 synthetic tests, Python 3.9 and 3.12). **Blocked on one real-data issue, see below.** |
| 3. Training | Not started. Needs the owner's OK after Phase 2. |
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

## Open issue that blocks Phase 2

On Studio Lab the zip was extracted (`data/raw/weedyrice/{RGB,Masks,Metadata}`), but
`prepare_weedyrice` has not produced a dataset yet. Facts so far:

- The Mendeley zip (about 5.3 GB) contains one folder with a nested `WeedyRice-RGBMS-DB.zip`.
  `rwd.download` now reads nested zips; `python -m rwd.download list <zip>` shows the layout.
- `Metadata/image_metadata.csv` columns: `original_filename, acquisition_date, time, latitude,
  longitude, altitude, camera_model, sensor_type, band`. There is also `Metadata/filename_mapping.csv`.
- The config already uses `filename_column: original_filename`, `group_column: acquisition_date`,
  `keep_columns: [altitude]`. Not yet known:
  - whether rows include multispectral bands (then filter to RGB rows, e.g. by `sensor_type`),
  - whether files in `RGB/` are named like `original_filename` or need `filename_mapping.csv`,
  - the mask file naming (`mask_name: "{stem}.png"` is a guess),
  - the number of images per date (to confirm the split is not too unbalanced).

Next step: inspect the real CSVs and file names, then extend `rwd/prepare_weedyrice.py`
(optional mapping CSV, optional row filter) with synthetic tests, build the dataset, check
overlays with `rwd.viz`, and report to the owner before Phase 3.

## PC setup (WSL2 Ubuntu, done by the next session with the owner)

1. Check the GPU from WSL: `nvidia-smi` must show the RTX 2080 SUPER. The NVIDIA driver is
   installed on Windows only, never inside Ubuntu.
2. `sudo apt update && sudo apt install -y git python3-venv python3-pip` (needs the owner's password).
3. `cd ~ && git clone https://github.com/Putinun-te/rice-weed-detection.git`
4. `bash ~/rice-weed-detection/scripts/setup_env.sh` (creates `.venv` and kernel `Python (rwd)`).
5. `.venv/bin/python -m pytest -q` must pass.
6. Data: `bash scripts/studiolab_data.sh` works on WSL too (needs about 7 GB free during download).
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
