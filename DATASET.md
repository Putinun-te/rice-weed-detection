# Datasets

WeedyRice-RGBMS-DB has been downloaded to `data/raw/` (outside git). No third-party images are committed to this repository.
None of these datasets are the author's own data.

Checked on 2026-11-11 from the public source pages (read-only, nothing downloaded).

## Primary candidate

### WeedyRice-RGBMS-DB

- Title: A dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice
- Authors: Van Hoa Nguyen, Cong-Doan Le, Minh-Tuyen Truong, Mai-Phung Bui, Thanh-Phong Le
- Source: [Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6), version 1, published 2025-07-21, DOI 10.17632/vt4s83pxx6.1
- Paper: Data in Brief 63:112237, DOI 10.1016/j.dib.2025.112237 ([PMC12670920](https://pmc.ncbi.nlm.nih.gov/articles/PMC12670920/))
- License: CC BY 4.0 (stated on Mendeley and in the paper). Attribution required: cite the dataset DOI and the paper.
- Content: 734 RGB JPG (delivered at 1280x960), 734x4-band multispectral TIF, 734 binary PNG masks (255 = weedy rice, 0 = background), overlays.
- Capture: DJI Mavic 3 Multispectral, 12 m and 20 m AGL, 09:30-15:30, 55-60 days after sowing. GSD not stated.
- Location and dates: An Giang Province, Vietnam. Thoaison District on 2024-06-02, 2024-06-04 and 2024-09-30; Longxuyen City on 2025-01-15.
- Metadata: `Metadata/image_metadata.csv` with acquisition date/time, GPS and altitude. No explicit field id.
- Splits: "sample" train/val/test text files are provided, but the split method is not stated. This project will build its own split by site and date instead.
- Limits for this project: only weedy rice is labelled (no `rice` class, no other weed types); masks, not boxes; Vietnam, not Thailand; one growth stage.

## Rejected or secondary

| Dataset | License | What the page shows | Decision |
|---|---|---|---|
| [Rice-weed-detection (RIG works, Roboflow)](https://universe.roboflow.com/rig-works/rice-weed-detection) | CC BY 4.0 | 30 images, 1 class `rice` only | Rejected: no weed class, too small |
| [UAV Open Dataset of Rice Paddies (PES, Roboflow)](https://universe.roboflow.com/pes-hdrvu/uav-open-dataset-of-rice-paddies) | CC BY 4.0 | 600 images, semantic segmentation, classes `RiceSeedling`, `Stressed` | Rejected: no weed class |
| [Weed Detection by UAV50 (Leas, Roboflow)](https://universe.roboflow.com/leas/weed-detection-by-uav50) | Public Domain | 946 images, 1 class `weeds`, no description, crop unknown | Secondary: train-only extra data, only after visual check that it is rice |
| [Weed Detection in Paddy fields (WEED, Roboflow)](https://universe.roboflow.com/weed-hdepn/weed-detection-in-paddy-fields) | CC BY 4.0 | 164 images, classes `weed`, `paady`, `paddy` (typo duplicate), no description | Secondary: train-only, needs class merge and viewpoint check |
| [rice-seedling-and-weed-dataset (Kaggle)](https://www.kaggle.com/datasets/ferox98/riceseedlingandweeddataset) | Unknown | No description | Rejected: license unknown |
| [crop and weed detection data with bounding boxes (Kaggle)](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) | not checked | Sesame crop, not rice | Rejected: wrong crop |

Roboflow pages give no source, capture method or split information. None of them can be used for validation or test.

## Used datasets

### WeedyRice-RGBMS-DB (v1) - the only dataset used so far

- Downloaded 2026-11-12 from Mendeley Data (DOI 10.17632/vt4s83pxx6.1), version 1, with
  `rwd.download` (`--include RGB Masks Metadata`; the multispectral TIFs were not extracted).
- License: CC BY 4.0. Attribution to use in any report or model card:
  Nguyen, V. H., Le, C.-D., Truong, M.-T., Bui, M.-P., Le, T.-P. (2025). *WeedyRice-RGBMS-DB:
  A dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice.*
  Mendeley Data, V1, doi:10.17632/vt4s83pxx6.1, and the Data in Brief paper doi:10.1016/j.dib.2025.112237.
- Verified content: 734 RGB JPG at 1280x960, 734 PNG masks (values 0 and 255 only, same stem as the
  JPG), no empty mask. Weedy rice covers 32% of the pixels on average (median 24%, max 90%), so the
  patches are large, not single plants. All RGB images come from camera FC6520.
- Metadata: `image_metadata.csv` has 5 rows per capture (1 RGB + 4 multispectral bands); only rows with
  `sensor_type == RGB` are used. Files on disk use the *standardized* names from `filename_mapping.csv`.
- Class: `weedy_rice` only (id 0). Converted to YOLO-seg polygons by `rwd.prepare_weedyrice`
  (mask vs re-drawn polygon IoU 0.997 mean, 0.954 min over 120 sampled images).
- Split by capture date and site (`configs/weedyrice.yaml`), whole images, no tiles:

  | Split | Dates | Site | Images | Polygons | Altitude |
  |---|---|---|---|---|---|
  | train | 2024-06-02, 2024-06-04 | Thoai Son (two fields about 1 km apart) | 446 | 3439 | 20 m |
  | val | 2024-09-30 | Thoai Son (same field as 2024-06-04, later season) | 41 | 388 | 20 m |
  | test | 2025-01-15 | Long Xuyen (about 7 km away) | 247 | 2516 | 20 m (183), 12 m (61), 21 m (3) |

  Validator: no label errors, no exact or near-duplicate images across splits, no group leakage.
- Known gaps: 12 m flights exist only in the test split, and the test field has denser, more
  fragmented weed patches than the training fields. The validation split is small (41 images).
