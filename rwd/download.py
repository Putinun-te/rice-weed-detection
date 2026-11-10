"""Download datasets into data/raw/ (outside git).

Keys are read from environment variables only (see .env.example). They are never printed or saved.

Mendeley Data (no key needed):
    python -m rwd.download mendeley --dataset-id vt4s83pxx6 --version 1 \
        --out data/raw/weedyrice --include RGB Masks Metadata

    The WeedyRice zip is about 5.3 GB because it also holds multispectral TIFs.
    Only the folders in --include are extracted, then the zip is deleted (unless --keep-zip).

Roboflow Universe (needs ROBOFLOW_API_KEY, `pip install roboflow`):
    python -m rwd.download roboflow --workspace leas --project weed-detection-by-uav50 --version 2 \
        --format yolov8 --out data/raw/uav50

Kaggle (needs KAGGLE_USERNAME and KAGGLE_KEY, `pip install kaggle`):
    python -m rwd.download kaggle --dataset owner/name --out data/raw/name
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import requests

MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"
CHUNK = 1024 * 1024


def require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Missing environment variable(s): {', '.join(missing)}. See .env.example.")
    return {n: os.environ[n] for n in names}


def _safe_target(out: Path, rel: PurePosixPath) -> Path:
    target = (out / Path(*rel.parts)).resolve()
    if out.resolve() not in target.parents and target != out.resolve():
        raise ValueError(f"unsafe path in zip: {rel}")
    return target


def _relative_from_include(name: str, include: set[str]) -> PurePosixPath | None:
    """Return the path starting at the first included folder, or None if not included."""
    parts = PurePosixPath(name).parts
    for i, part in enumerate(parts):
        if part in include:
            return PurePosixPath(*parts[i:])
    return None


def extract_selected(zip_path: Path, out: Path, include: list[str] | None) -> int:
    """Extract members under the included folder names. Inner zips named like a folder (RGB.zip) are opened too."""
    inc = set(include or [])
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            pure = PurePosixPath(info.filename)
            if pure.suffix.lower() == ".zip" and (not inc or pure.stem in inc):
                inner = out / f"_inner_{pure.name}"
                with zf.open(info) as src, open(inner, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                count += extract_selected_into(inner, out / pure.stem)
                inner.unlink()
                continue
            rel = _relative_from_include(info.filename, inc) if inc else pure
            if rel is None:
                continue
            target = _safe_target(out, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def extract_selected_into(zip_path: Path, out: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = _safe_target(out, PurePosixPath(info.filename))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def download_mendeley(dataset_id: str, version: int, out: Path, include: list[str], max_gb: float, keep_zip: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{dataset_id}-{version}.zip"
    url = MENDELEY_ZIP_URL.format(dataset_id=dataset_id, version=version)

    if not zip_path.exists():
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            size = int(r.headers.get("Content-Length", 0))
            gb = size / 1024**3
            free = shutil.disk_usage(out).free / 1024**3
            print(f"zip size: {gb:.2f} GB, free disk: {free:.2f} GB")
            if size and gb > max_gb:
                raise SystemExit(f"zip is {gb:.2f} GB, above --max-gb {max_gb}. Raise --max-gb to continue.")
            if size and free < gb * 1.3:
                raise SystemExit("not enough free disk for the zip plus extracted files")
            part = zip_path.with_suffix(".part")
            done = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if size and done % (256 * CHUNK) < CHUNK:
                        print(f"  {done / size:6.1%}", flush=True)
            part.rename(zip_path)
    else:
        print(f"using existing {zip_path}")

    n = extract_selected(zip_path, out, include)
    print(f"extracted {n} file(s) into {out}")
    if n == 0:
        with zipfile.ZipFile(zip_path) as zf:
            top = sorted({PurePosixPath(i.filename).parts[0] for i in zf.infolist()})[:20]
        raise SystemExit(f"nothing matched --include {include}. Top-level entries: {top}. Zip kept.")
    if not keep_zip:
        zip_path.unlink()
        print("deleted zip to save disk")


def download_roboflow(workspace: str, project: str, version: int, fmt: str, out: Path) -> None:
    key = require_env("ROBOFLOW_API_KEY")["ROBOFLOW_API_KEY"]
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow")
    Roboflow(api_key=key).workspace(workspace).project(project).version(version).download(fmt, location=str(out))


def download_kaggle(dataset: str, out: Path) -> None:
    require_env("KAGGLE_USERNAME", "KAGGLE_KEY")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(out), unzip=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)

    m = sub.add_parser("mendeley")
    m.add_argument("--dataset-id", required=True)
    m.add_argument("--version", type=int, default=1)
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--include", nargs="*", default=[], help="folder names to extract (default: everything)")
    m.add_argument("--max-gb", type=float, default=6.0)
    m.add_argument("--keep-zip", action="store_true")

    r = sub.add_parser("roboflow")
    r.add_argument("--workspace", required=True)
    r.add_argument("--project", required=True)
    r.add_argument("--version", type=int, required=True)
    r.add_argument("--format", default="yolov8")
    r.add_argument("--out", required=True, type=Path)

    k = sub.add_parser("kaggle")
    k.add_argument("--dataset", required=True)
    k.add_argument("--out", required=True, type=Path)

    args = ap.parse_args(argv)
    if args.source == "mendeley":
        download_mendeley(args.dataset_id, args.version, args.out, args.include, args.max_gb, args.keep_zip)
    elif args.source == "roboflow":
        download_roboflow(args.workspace, args.project, args.version, args.format, args.out)
    else:
        download_kaggle(args.dataset, args.out)
    print("Record source, license and attribution in DATASET.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
