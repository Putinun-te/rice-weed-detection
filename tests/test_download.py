from __future__ import annotations

import io
import zipfile

import pytest

from rwd.download import extract_selected, list_zip, require_env


def _zip(path, files: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_extracts_only_included_folders(tmp_path):
    z = tmp_path / "d.zip"
    _zip(z, {"Top/RGB/a.jpg": b"a", "Top/Multispectral/a.tif": b"big", "Top/Metadata/m.csv": b"x"})
    out = tmp_path / "out"
    out.mkdir()
    assert extract_selected(z, out, ["RGB", "Metadata"]) == 2
    assert (out / "RGB/a.jpg").read_bytes() == b"a"
    assert not (out / "Multispectral").exists()


def test_opens_inner_folder_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("a.png", b"m")
    z = tmp_path / "d.zip"
    _zip(z, {"Top/Masks.zip": inner.getvalue(), "Top/Multispectral.zip": inner.getvalue()})
    out = tmp_path / "out"
    out.mkdir()
    assert extract_selected(z, out, ["Masks"]) == 1
    assert (out / "Masks/a.png").exists()
    assert not (out / "Multispectral").exists()


def _weedyrice_like(tmp_path, compression):
    """Same shape as the real Mendeley download: outer zip -> folder -> WeedyRice-RGBMS-DB.zip -> folders."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("WeedyRice-RGBMS-DB/RGB/DJI_0001.JPG", b"rgb")
        zf.writestr("WeedyRice-RGBMS-DB/Masks/DJI_0001.png", b"mask")
        zf.writestr("WeedyRice-RGBMS-DB/Metadata/image_metadata.csv", b"csv")
        zf.writestr("WeedyRice-RGBMS-DB/Multispectral/DJI_0001_NIR.TIF", b"big" * 1000)
    z = tmp_path / "vt4s83pxx6-1.zip"
    with zipfile.ZipFile(z, "w", compression) as zf:
        zf.writestr("A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB.zip", inner.getvalue())
    return z


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_looks_inside_nested_dataset_zip(tmp_path, compression):
    z = _weedyrice_like(tmp_path, compression)
    out = tmp_path / "out"
    assert extract_selected(z, out, ["RGB", "Masks", "Metadata"]) == 3
    assert (out / "RGB/DJI_0001.JPG").read_bytes() == b"rgb"
    assert (out / "Metadata/image_metadata.csv").exists()
    assert not (out / "Multispectral").exists()
    assert not list(out.glob("_tmp_*"))  # temp copy of a compressed inner zip is removed


def test_list_zip_shows_nested_stored_zip(tmp_path):
    z = _weedyrice_like(tmp_path, zipfile.ZIP_STORED)
    text = "\n".join(list_zip(z))
    assert "WeedyRice-RGBMS-DB.zip  (nested zip" in text and "stored" in text
    assert "1 files  WeedyRice-RGBMS-DB/RGB/" in text


def test_rejects_zip_slip(tmp_path):
    z = tmp_path / "d.zip"
    _zip(z, {"../evil.txt": b"x"})
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        extract_selected(z, out, None)


def test_missing_env_key_exits_without_value(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ROBOFLOW_API_KEY"):
        require_env("ROBOFLOW_API_KEY")
