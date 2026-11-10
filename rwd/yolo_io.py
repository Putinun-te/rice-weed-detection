"""Small helpers for reading and writing YOLO-format datasets (detect and segment).

Layout used by Ultralytics:
    <root>/images/<split>/name.jpg
    <root>/labels/<split>/name.txt

Label line formats:
    detect:  class cx cy w h            (normalized 0..1)
    segment: class x1 y1 x2 y2 ... xn yn (normalized 0..1, at least 3 points)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
TASKS = ("detect", "segment")


@dataclass
class LabelLine:
    class_id: int
    coords: list[float]


@dataclass
class LabelProblem:
    line_no: int
    text: str
    reason: str


@dataclass
class DataConfig:
    root: Path
    names: dict[int, str]
    splits: dict[str, Path]  # split name -> images directory


def load_data_yaml(path: str | Path) -> DataConfig:
    """Read a data.yaml. A relative `path:` is resolved against the yaml's folder.

    Note: Ultralytics resolves a relative `path:` against its own datasets dir instead,
    so the files this project writes always use an absolute path.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    root = Path(cfg.get("path") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    names = cfg["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}

    splits = {s: root / cfg[s] for s in SPLITS if cfg.get(s)}
    return DataConfig(root=root, names=names, splits=splits)


def write_data_yaml(path: str | Path, root: str | Path, names: dict[int, str], splits: list[str]) -> None:
    """Write a data.yaml with an absolute root path and images/<split> folders."""
    data = {
        "path": str(Path(root).resolve()),
        **{s: f"images/{s}" for s in splits},
        "names": {int(k): v for k, v in names.items()},
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path) -> Path:
    """Same rule as Ultralytics: replace the last 'images' folder with 'labels', extension .txt."""
    parts = list(image_path.parts)
    idx = [i for i, p in enumerate(parts) if p == "images"]
    if not idx:
        raise ValueError(f"No 'images' folder in path: {image_path}")
    parts[idx[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_label_file(path: Path, task: str, num_classes: int) -> tuple[list[LabelLine], list[LabelProblem]]:
    """Parse one label file. Returns good lines and a list of problems (never raises on bad content)."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    lines: list[LabelLine] = []
    problems: list[LabelProblem] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            problems.append(LabelProblem(line_no, text, "not numeric"))
            continue

        reason = _check_line(class_id, coords, task, num_classes)
        if reason:
            problems.append(LabelProblem(line_no, text, reason))
        else:
            lines.append(LabelLine(class_id, coords))
    return lines, problems


def _check_line(class_id: int, coords: list[float], task: str, num_classes: int) -> str | None:
    if not 0 <= class_id < num_classes:
        return f"class id {class_id} not in names (0..{num_classes - 1})"
    if task == "detect":
        if len(coords) != 4:
            return f"detect line needs 4 values, got {len(coords)}"
        if coords[2] <= 0 or coords[3] <= 0:
            return "box width/height must be > 0"
    else:
        if len(coords) < 6 or len(coords) % 2:
            return f"segment line needs an even number of values >= 6, got {len(coords)}"
    if any(v < -1e-6 or v > 1 + 1e-6 for v in coords):
        return "coordinate outside 0..1"
    return None
