# Rice-Weed Detection

Weed detection in rice fields from UAV RGB images, with Ultralytics YOLO trained on
Amazon SageMaker Studio Lab and exported to NCNN/ONNX for edge boards.

Status: planning (Phase 1). No model, no results. See [docs/PLAN.md](docs/PLAN.md) and [DATASET.md](DATASET.md).

## Data tooling (Phase 2)

```bash
uv venv .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/pytest -q                                   # synthetic tests, no real data

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
