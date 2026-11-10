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

Look inside a zip without extracting (also inside nested zips that are stored, not compressed):
    python -m rwd.download list data/raw/weedyrice/vt4s83pxx6-1.zip
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import struct
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import requests

MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"
CHUNK = 1024 * 1024
MAX_ZIP_DEPTH = 3


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


class _FileSlice(io.RawIOBase):
    """Read-only, seekable window [start, start + length) of a file on disk.

    Lets zipfile open a zip that is stored (not compressed) inside another zip
    without copying gigabytes to disk first.
    """

    def __init__(self, path: Path, start: int, length: int):
        self._f = open(path, "rb")
        self._start, self._length, self._pos = start, length, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._length}[whence]
        self._pos = max(0, min(self._length, base + offset))
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._length - self._pos)
        if n <= 0:
            return 0
        self._f.seek(self._start + self._pos)
        data = self._f.read(n)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        self._f.close()
        super().close()


def _member_data_offset(base_path: Path, base_start: int, info: zipfile.ZipInfo) -> int:
    """Absolute file offset where a member's bytes start (after its local header)."""
    with open(base_path, "rb") as f:
        f.seek(base_start + info.header_offset)
        header = f.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return base_start + info.header_offset + 30 + name_len + extra_len


@contextmanager
def _open_inner_zip(zf: zipfile.ZipFile, info: zipfile.ZipInfo, base_path: Path, base_start: int, tmp_dir: Path):
    """Open a zip member as a ZipFile. Yields (ZipFile, path, start) for further nesting.

    Stored member: read in place (no extra disk). Compressed member: copy to a temp file first.
    """
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted zip member not supported: {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        start = _member_data_offset(base_path, base_start, info)
        with zipfile.ZipFile(_FileSlice(base_path, start, info.file_size)) as inner:
            yield inner, base_path, start
        return

    free = shutil.disk_usage(tmp_dir).free
    if free < info.file_size * 1.1:
        raise SystemExit(f"need {info.file_size / 1024**3:.1f} GB free to unpack {info.filename}, have {free / 1024**3:.1f} GB")
    tmp = tmp_dir / f"_tmp_{PurePosixPath(info.filename).name}"
    print(f"unpacking nested zip {info.filename} ({info.file_size / 1024**3:.2f} GB) to a temp file")
    try:
        with zf.open(info) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        with zipfile.ZipFile(tmp) as inner:
            yield inner, tmp, 0
    finally:
        tmp.unlink(missing_ok=True)


def extract_selected(zip_path: Path, out: Path, include: list[str] | None) -> int:
    """Extract members under the included folder names, looking inside nested zips.

    - a.zip/Top/RGB/x.jpg with include RGB  -> out/RGB/x.jpg
    - a.zip/Top/RGB.zip                      -> out/RGB/... (whole inner zip)
    - a.zip/Top/Data.zip/RGB/x.jpg           -> out/RGB/x.jpg (looks inside Data.zip)
    """
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        return _extract_from(zf, Path(zip_path), 0, out, out, set(include or []), depth=0)


def _extract_from(zf, base_path, base_start, out, tmp_dir, inc, depth) -> int:
    count = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        pure = PurePosixPath(info.filename)
        if pure.suffix.lower() == ".zip":
            if depth >= MAX_ZIP_DEPTH:
                continue
            if not inc or pure.stem in inc:
                sub_out, sub_inc = _safe_target(out, PurePosixPath(pure.stem)), set()
            else:
                sub_out, sub_inc = out, inc
            with _open_inner_zip(zf, info, base_path, base_start, tmp_dir) as (inner, ipath, istart):
                count += _extract_from(inner, ipath, istart, sub_out, tmp_dir, sub_inc, depth + 1)
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


def list_zip(zip_path: Path, levels: int = 3) -> list[str]:
    """Folder summary of a zip, including stored nested zips, as printable lines."""
    lines: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        _list_from(zf, Path(zip_path), 0, levels, "", lines, depth=0)
    return lines


def _list_from(zf, base_path, base_start, levels, indent, lines, depth) -> None:
    counts: Counter = Counter()
    for info in zf.infolist():
        if info.is_dir():
            continue
        pure = PurePosixPath(info.filename)
        if pure.suffix.lower() == ".zip":
            how = "stored" if info.compress_type == zipfile.ZIP_STORED else "compressed"
            lines.append(f"{indent}{info.filename}  (nested zip, {info.file_size / 1024**3:.2f} GB, {how})")
            if how == "stored" and depth < MAX_ZIP_DEPTH:
                start = _member_data_offset(base_path, base_start, info)
                with zipfile.ZipFile(_FileSlice(base_path, start, info.file_size)) as inner:
                    _list_from(inner, base_path, start, levels, indent + "    ", lines, depth + 1)
            elif how == "compressed":
                lines.append(f"{indent}    (contents not listed: compressed nested zip)")
            continue
        counts["/".join(pure.parts[: min(levels, len(pure.parts) - 1)]) or "."] += 1
    for folder, n in sorted(counts.items()):
        lines.append(f"{indent}{n:>6} files  {folder}/")


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
        print("\n".join(list_zip(zip_path)[:60]))
        raise SystemExit(f"nothing matched --include {include}. Zip contents are listed above. Zip kept.")
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

    ls = sub.add_parser("list", help="show folders inside a zip, including stored nested zips (no extraction)")
    ls.add_argument("zip", type=Path)

    args = ap.parse_args(argv)
    if args.source == "list":
        print("\n".join(list_zip(args.zip)))
        return 0
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
