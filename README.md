# Rice-Weed Detection

Weed detection in rice fields from UAV RGB images, with Ultralytics YOLO trained on
Amazon SageMaker Studio Lab and exported to NCNN/ONNX for edge boards.

Status: Phase 2 (data tooling). No model, no results. Next steps: [docs/HANDOFF.md](docs/HANDOFF.md). See [docs/PLAN.md](docs/PLAN.md) and [DATASET.md](DATASET.md).

## On SageMaker Studio Lab (CPU runtime is enough for data)

Studio Lab's default env is Python 3.9 with pinned packages, so the project uses its own `.venv`
and a Jupyter kernel named `Python (rwd)`. Code supports Python 3.9+.

```bash
cd ~ && git clone https://github.com/Putinun-te/rice-weed-detection.git
bash ~/rice-weed-detection/scripts/setup_env.sh
```

Start with `notebooks/00_overview.ipynb` to see what has been done so far.
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
