from __future__ import annotations

from scripts.nbstrip import strip


def test_strip_removes_outputs_and_counts():
    nb = {"cells": [
        {"cell_type": "code", "source": ["x = 1"], "outputs": [{"data": {"image/png": "abc"}}], "execution_count": 3},
        {"cell_type": "markdown", "source": ["text"]},
    ]}
    out = strip(nb)
    assert out["cells"][0]["outputs"] == []
    assert out["cells"][0]["execution_count"] is None
    assert out["cells"][0]["source"] == ["x = 1"]
    assert out["cells"][1] == {"cell_type": "markdown", "source": ["text"]}
