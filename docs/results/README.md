# Result files

Source files behind the numbers in [`../RESULTS.md`](../RESULTS.md), copied from
`runs/weedyrice_yolo26n-seg_1280/` on 2026-11-13 with local paths made relative. Numbers only, no images.

| File | What it is |
|---|---|
| `eval_test.json` | The single evaluation on the held-out test split (247 images, Long Xuyen, 2025-01-15) at confidence 0.25: pixel precision / recall / IoU overall, by flight height and per image, plus instance metrics. |
| `train_env.json` | Training settings, GPU, library versions and commit of the chosen run. |
| `export.json` | Exported formats and file sizes. |
| `benchmark_cpu.json` | CPU speed on the desktop PC (Intel Core i9-9900KS). The MacBook M4 numbers in `RESULTS.md` were measured on the Mac; that file was not copied here. |

| `val_comparison.json` | Validation-split scores of all four trained models, per confidence (pixel) plus instance metrics. For comparing models only, see [`../MODELS.md`](../MODELS.md). |
| `val_comparison.png`, `training_curves_val.png` | Charts of the validation comparison (numbers only, no dataset images). |
| `test_result.png` | Chart of `eval_test.json`: the chosen model's test result by flight height. |

Validation scores were used for model selection and are not a held-out result. `train_env.json` has them removed.
Regenerate the JSON and charts with `.venv/bin/python scripts/make_model_report.py`.

## Data credit

The model was trained and evaluated on **WeedyRice-RGBMS-DB** (RGB images and masks only), licensed CC BY 4.0:

Nguyen, V. H., Le, C.-D., Truong, M.-T., Bui, M.-P., Le, T.-P. (2025). *WeedyRice-RGBMS-DB: A dataset of
aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice.* Mendeley Data, V1,
doi:[10.17632/vt4s83pxx6.1](https://doi.org/10.17632/vt4s83pxx6.1). Data in Brief paper:
doi:[10.1016/j.dib.2025.112237](https://doi.org/10.1016/j.dib.2025.112237).

The dataset is third-party work and is not part of this repository. The image names in `eval_test.json`
are the dataset's file names. Changes made: masks converted to YOLO polygons, images split by capture date
and site. Full record: [`../../DATASET.md`](../../DATASET.md).
