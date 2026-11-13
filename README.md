# Rice-Weed Detection

Weedy-rice detection in rice fields from UAV RGB images, with Ultralytics YOLO (segmentation),
exported to ONNX/NCNN, with detection-only inference that logs to CSV.

Status: finished on public data (Track B). Results, limits and how to reproduce: [docs/RESULTS.md](docs/RESULTS.md).
Work log: [docs/HANDOFF.md](docs/HANDOFF.md). Plan: [docs/PLAN.md](docs/PLAN.md). Data: [DATASET.md](DATASET.md).

Held-out test result (247 images from a site and date not used for training, `yolo26n-seg`, confidence 0.25):
pixel precision 0.889, pixel recall 0.697, pixel IoU 0.641 for weedy rice. The goal of recall around 0.80 was
not met. No field, drone or edge-board validation. Detection only: never connect the output to an actuator.

## Data credit

Trained and evaluated on **WeedyRice-RGBMS-DB** (CC BY 4.0), a third-party dataset that is not part of this
repository: Nguyen, V. H., Le, C.-D., Truong, M.-T., Bui, M.-P., Le, T.-P. (2025). *WeedyRice-RGBMS-DB: A
dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice.* Mendeley Data,
V1, doi:[10.17632/vt4s83pxx6.1](https://doi.org/10.17632/vt4s83pxx6.1); Data in Brief,
doi:[10.1016/j.dib.2025.112237](https://doi.org/10.1016/j.dib.2025.112237).
One optional experiment also used CoFly-WeedDB (Krestenitis et al., 2022, CC BY 4.0); the released model does not.
Full records: [DATASET.md](DATASET.md). Result files: [docs/results/](docs/results/).

Trained model (PyTorch, ONNX, NCNN) and the test result file: [release v1.0.0](https://github.com/Putinun-te/rice-weed-detection/releases/tag/v1.0.0).

## On SageMaker Studio Lab (CPU runtime is enough for data)

Studio Lab's default env is Python 3.9 with pinned packages, so the project uses its own `.venv`
and a Jupyter kernel named `Python (rwd)`. Code supports Python 3.9+.

```bash
cd ~ && git clone https://github.com/Putinun-te/rice-weed-detection.git
bash ~/rice-weed-detection/scripts/setup_env.sh
```

Start with `notebooks/00_overview.ipynb` to see what has been done so far.
Run any notebook from a terminal with `.venv/bin/python scripts/run_notebook.py notebooks/<name>.ipynb`.
Then either open `notebooks/01_data_setup.ipynb` and pick **Kernel → Change Kernel → Python (rwd)**,
or run everything from a terminal in the background:

```bash
nohup bash ~/rice-weed-detection/scripts/studiolab_data.sh > ~/rice-weed-detection/studiolab_data.log 2>&1 &
```

## Data tooling (Phase 2)

```bash
bash scripts/setup_env.sh                             # or: uv venv .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m pytest -q                         # synthetic tests, no real data

# 1. download WeedyRice (about 5.3 GB zip; only RGB, Masks, Metadata are kept)
python -m rwd.download mendeley --dataset-id vt4s83pxx6 --version 1 \
    --out data/raw/weedyrice --include RGB Masks Metadata

# 2. look at the raw files (config values are already confirmed for WeedyRice v1)
python -m rwd.prepare_weedyrice --config configs/weedyrice.yaml --inspect

# 3. build the YOLO-seg dataset (split by capture date) and validate it
python -m rwd.prepare_weedyrice --config configs/weedyrice.yaml

# validate any YOLO dataset
python -m rwd.validate data/weedyrice_seg/data.yaml --task segment \
    --groups data/weedyrice_seg/groups.csv --near-dup 4 --out report.json
```

Detection only. This project must never be connected to a sprayer, solenoid or any actuator.

## License

Code: [AGPL-3.0](LICENSE), matching Ultralytics YOLO which this project uses.
Datasets are not part of this repository and keep their own licenses; see [DATASET.md](DATASET.md).

## Training (Phase 3)

```bash
.venv/bin/python -m rwd.train --config configs/train.yaml --check   # GPU and dataset check
.venv/bin/python -m rwd.train --config configs/train.yaml --auto    # start, continue a stopped run, or skip if finished
```

The same steps are in `notebooks/02_train.ipynb`. Settings live in `configs/train.yaml`.
Training evaluates on the val split only; the test split is used once, in Phase 4.

## Evaluation (Phase 4 tooling)

```bash
# val: choose the model and the confidence threshold (not a publishable result)
.venv/bin/python -m rwd.evaluate --weights runs/<name>/weights/best.pt --split val --conf 0.05 0.1 0.15 0.25 0.4 0.5
```

Headline metric: pixel precision, recall and IoU, with the instance metrics next to it
(`docs/PLAN.md`). The test split is evaluated once, with `--split test --conf <chosen> --final`.
The same steps are in `notebooks/03_evaluate.ipynb`.

## Export and inference (Phase 5)

```bash
.venv/bin/python -m pip install -r requirements-export.txt
.venv/bin/python -m rwd.export --weights runs/<name>/weights/best.pt            # ONNX + NCNN, input 960x1280
.venv/bin/python -m rwd.benchmark --weights runs/<name>/weights/best.onnx --images <folder>
.venv/bin/python -m rwd.infer --weights runs/<name>/weights/best.onnx --source <folder> --out detections.csv
```

The same steps are in `notebooks/05_export_infer.ipynb`. `rwd.infer` only writes a CSV log (and optional
overlay images): detection only, never connected to an actuator. CPU timings describe the computer they
were measured on; nothing has been run on a Raspberry Pi (`docs/EDGE_CHECKLIST.md` is untested).

## What is used and what is optional

| Part | Status |
|---|---|
| `configs/train.yaml`, run `weedyrice_yolo26n-seg_1280` | **Used.** The chosen model, the only one evaluated on test. |
| Notebooks 00, 01, 02, 03, 05 | **Used.** Overview, data, training, evaluation, export and inference. |
| `configs/train_s.yaml`, `configs/train_s_aug.yaml` | Optional. Larger model and stronger augmentation; no gain on val. |
| `configs/cofly.yaml`, `configs/train_cofly.yaml`, `rwd/add_extra.py`, notebook 04 | Optional. CoFly-WeedDB as an extra class; no gain on val. |
