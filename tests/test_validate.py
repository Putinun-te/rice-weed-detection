import shutil

from rwd.validate import main, validate
from rwd.yolo_io import write_data_yaml

from conftest import add_sample


def test_clean_dataset_passes(clean_dataset):
    r = validate(clean_dataset / "data.yaml", "segment", clean_dataset / "groups.csv", near_dup_max_distance=0)
    assert r["ok"], r["errors"]
    assert r["splits"]["train"]["images"] == 2
    assert r["splits"]["train"]["instances"] == {"weedy_rice": 2}
    assert r["leakage"]["groups_per_split"]["test"] == ["2025-01-15"]


def test_class_with_zero_train_instances_fails(clean_dataset):
    write_data_yaml(clean_dataset / "data.yaml", clean_dataset, {0: "weedy_rice", 1: "rice"}, ["train", "val", "test"])
    r = validate(clean_dataset / "data.yaml", "segment")
    assert not r["ok"]
    assert any("'rice'" in e and "zero instances in train" in e for e in r["errors"])


def test_empty_and_missing_labels_are_counted(clean_dataset):
    add_sample(clean_dataset, "train", "empty", 10, label="")
    add_sample(clean_dataset, "train", "nolabel", 11, label=None)
    r = validate(clean_dataset / "data.yaml", "segment")
    assert r["splits"]["train"]["empty_labels"] == 1
    assert r["splits"]["train"]["missing_labels"] == 1
    assert r["ok"]  # both are allowed (background), only reported


def test_bad_label_line_fails(clean_dataset):
    (clean_dataset / "labels/train/a.txt").write_text("0 0.5 0.5\n")
    r = validate(clean_dataset / "data.yaml", "segment")
    assert not r["ok"]
    assert r["splits"]["train"]["bad_lines"] == 1


def test_duplicate_image_across_splits_fails(clean_dataset):
    shutil.copy(clean_dataset / "images/train/a.png", clean_dataset / "images/test/a_copy.png")
    shutil.copy(clean_dataset / "labels/train/a.txt", clean_dataset / "labels/test/a_copy.txt")
    r = validate(clean_dataset / "data.yaml", "segment")
    assert not r["ok"]
    assert r["duplicates"]["across_splits"] == [["images/train/a.png", "images/test/a_copy.png"]]


def test_near_duplicate_is_warning(clean_dataset):
    shutil.copy(clean_dataset / "images/train/a.png", clean_dataset / "images/val/a_copy.png")
    r = validate(clean_dataset / "data.yaml", "segment", near_dup_max_distance=0)
    assert r["duplicates"]["near_across_splits"][0]["distance"] == 0


def test_group_leakage_fails(clean_dataset):
    csv = clean_dataset / "groups.csv"
    csv.write_text(csv.read_text().replace("images/test/d.png,2025-01-15", "images/test/d.png,2024-06-02"))
    r = validate(clean_dataset / "data.yaml", "segment", csv)
    assert not r["ok"]
    assert r["leakage"]["groups_in_multiple_splits"] == {"2024-06-02": ["test", "train"]}


def test_image_without_group_fails(clean_dataset):
    add_sample(clean_dataset, "val", "extra", 12)
    r = validate(clean_dataset / "data.yaml", "segment", clean_dataset / "groups.csv")
    assert not r["ok"]
    assert r["leakage"]["images_without_group"] == 1


def test_cli_exit_code_and_json(clean_dataset, tmp_path):
    out = tmp_path / "report.json"
    code = main([str(clean_dataset / "data.yaml"), "--task", "segment", "--groups", str(clean_dataset / "groups.csv"), "--out", str(out)])
    assert code == 0 and out.exists()
