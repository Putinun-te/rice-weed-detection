# Datasets

No dataset has been downloaded or used yet. No third-party images are committed to this repository.
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

## Possible later use (not downloaded into this project, not used)

Checked on 2026-11-13: licenses from the source pages, contents from copies the owner already had
on an external drive (listed without extracting).

### Rice Field Weed BD (Bangladesh)

- Title: A comprehensive dataset of rice field weed detection from Bangladesh
- Source: [Mendeley Data](https://data.mendeley.com/datasets/mt72bmxz73/4), version 4, published 2024-09-04, DOI 10.17632/mt72bmxz73.4
- Contributors: Sawkat Ali, Mohammad Rifat Ahmmad Rashid, Tasnim Hossain Orpa, Md Ahsan Kabir, Md Kamrul,
  Sayam Hossain Bhuiyan Aumy, Mehedi Hasan Mridha, Imam Hossain Sajeeb, Mohammad Manzurul Islam, Taskeed Jabid
- License: CC BY 4.0 (stated on the Mendeley page)
- Content: 3632 RGB JPG in 11 class folders, one weed species per folder (for example Cyperus ochraceus,
  Fimbristylis littoralis, Panicum repens, Paspalum scrobiculatum, Ipomoea aquatica, Marsilea minuta).
- Capture: smartphone camera, rice fields of Narsingdi and Munshiganj, Bangladesh. Not UAV. Dates not stated.
- Annotation: class folders only. No boxes or masks.
- Possible use: a later weed-type classifier (grass / sedge / broadleaf). It cannot train or evaluate
  the UAV segmentation model.

### CoFly-WeedDB

- Source: [Zenodo](https://zenodo.org/records/6697343), version 1.0, published 2022-06-23, DOI 10.5281/zenodo.6697343
- Creators: Marios Krestenitis, Emmanuel K. Raptis, Athanasios Ch. Kapoutsis, Konstantinos Ioannidis,
  Elias B. Kosmatopoulos, Stefanos Vrochidis, Ioannis Kompatsiaris
- Paper: Data in Brief 45:108575, DOI 10.1016/j.dib.2022.108575
- License: Creative Commons Attribution 4.0 International (stated on the Zenodo record)
- Content: 201 annotated RGB images (1280x720) with segmentation masks; 3 weed classes
  (Johnson grass, field bindweed, purslane).
- Capture: DJI Phantom 4 Pro, 5 m altitude, cotton field in Larissa, Greece. File names carry GPS and
  time; all are from 2019-07-03.
- Limits: cotton, not rice; one field and one date, so no split by field/date is possible.
- Possible use: train-only extra data or a second test of the mask pipeline. Never validation or test.

## Used datasets

None yet. For each dataset used, record: name, URL, version or download date, license,
required attribution text, classes, image count, and which split it is used in.
