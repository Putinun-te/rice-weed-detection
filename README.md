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

# 2. look at the raw files, then fill the TODO values in configs/weedyrice.yaml
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
