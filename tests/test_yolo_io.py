from pathlib import Path

import pytest

from rwd.yolo_io import label_path_for, load_data_yaml, parse_label_file, write_data_yaml


def test_label_path_uses_last_images_folder():
    p = Path("/x/images/data/images/train/a.jpg")
    assert label_path_for(p) == Path("/x/images/data/labels/train/a.txt")


def test_label_path_without_images_folder_fails():
    with pytest.raises(ValueError):
        label_path_for(Path("/x/train/a.jpg"))


def test_data_yaml_round_trip(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    write_data_yaml(yaml_path, tmp_path, {0: "weedy_rice"}, ["train", "val"])
    cfg = load_data_yaml(yaml_path)
    assert cfg.root == tmp_path.resolve()
    assert cfg.names == {0: "weedy_rice"}
    assert cfg.splits == {"train": tmp_path.resolve() / "images/train", "val": tmp_path.resolve() / "images/val"}


def test_parse_segment_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2\n"  # ok
        "1 0.1 0.1 0.2 0.1 0.2 0.2\n"  # unknown class
        "0 0.1 0.1 0.2\n"  # too few values
        "0 0.1 0.1 0.2 0.1 1.5 0.2\n"  # out of range
        "x 0.1\n"  # not numeric
        "\n"
    )
    lines, problems = parse_label_file(f, "segment", num_classes=1)
    assert len(lines) == 1
    assert [p.line_no for p in problems] == [2, 3, 4, 5]


def test_parse_detect_rejects_zero_size_box(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.0 0.1\n0 0.5 0.5 0.1 0.1\n")
    lines, problems = parse_label_file(f, "detect", num_classes=1)
    assert len(lines) == 1 and len(problems) == 1
