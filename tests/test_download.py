from __future__ import annotations

import io
import zipfile

import pytest

from rwd.download import extract_selected, require_env


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
