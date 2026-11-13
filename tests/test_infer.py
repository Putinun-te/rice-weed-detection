"""CSV logging helpers. No model, no torch."""

from __future__ import annotations

import csv

from rwd.benchmark import peak_rss_mb, summarize
from rwd.export import export_record, size_mb
from rwd.infer import FIELDS, parse_gps, rows_for_image, write_csv

NAME = "DJI_DateTime_2025_01_15_10_02_0007_lat_10.3791_lon_105.4412_alt_12.03m.JPG"


def test_parse_gps_from_weedyrice_name_and_missing():
    assert parse_gps(NAME) == {"lat": 10.3791, "lon": 105.4412, "alt_m": 12.03}
    assert parse_gps("lat_-3.5_lon_-60.25.jpg") == {"lat": -3.5, "lon": -60.25, "alt_m": ""}
    assert parse_gps("IMG_0001.jpg") == {"lat": "", "lon": "", "alt_m": ""}


def test_rows_one_per_detection_with_class_name_kept():
    dets = [{"class": "weedy_rice", "confidence": 0.91234, "box": (1.04, 2, 30, 40.96), "mask_area_px": 500},
            {"class": "other_weed", "confidence": 0.4, "box": (5, 5, 9, 9)}]
    rows = rows_for_image(NAME, 1280, 960, dets, "best.pt", 0.25, "2026-11-13T00:00:00+00:00")
    assert [r["detection"] for r in rows] == [1, 2]
    assert rows[0]["class"] == "weedy_rice" and rows[1]["class"] == "other_weed"  # never relabelled
    assert rows[0]["confidence"] == 0.9123 and rows[0]["x1"] == 1.0 and rows[0]["y2"] == 41.0
    assert rows[0]["lat"] == 10.3791 and rows[1]["mask_area_px"] == ""
    assert set(rows[0]) == set(FIELDS)


def test_image_without_detection_is_still_logged(tmp_path):
    rows = rows_for_image("a.jpg", 10, 10, [], "best.onnx", 0.25, "t")
    assert len(rows) == 1 and rows[0]["detection"] == 0 and rows[0]["class"] == ""
    out = tmp_path / "log/d.csv"
    write_csv(out, rows)
    write_csv(out, rows, append=True)
    back = list(csv.DictReader(open(out)))
    assert len(back) == 2 and list(back[0]) == FIELDS and back[0]["detection"] == "0"


def test_size_and_benchmark_summary(tmp_path):
    d = tmp_path / "m_ncnn_model"
    d.mkdir()
    (d / "a.bin").write_bytes(b"x" * 1024 * 1024)
    (d / "a.param").write_bytes(b"x" * 1024 * 1024)
    assert size_mb(d) == 2.0 and export_record("ncnn", d, [960, 1280])["imgsz_hw"] == [960, 1280]
    s = summarize([100.0, 200.0, 300.0, 400.0])
    assert s["median_ms"] == 250.0 and s["fps_median"] == 4.0 and s["images"] == 4


def test_peak_memory_units_differ_between_linux_and_macos():
    assert peak_rss_mb(1024 * 1024, "linux") == 1024.0      # kB
    assert peak_rss_mb(1024 * 1024 * 1024, "darwin") == 1024.0  # bytes
