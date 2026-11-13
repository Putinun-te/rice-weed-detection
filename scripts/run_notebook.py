"""Run a notebook top to bottom in the "rwd" kernel and save the outputs in place.

    .venv/bin/python scripts/run_notebook.py notebooks/00_overview.ipynb

Same as "Run All" in VS Code. Outputs stay local: the nbstrip git filter keeps them out of commits.
"""
from __future__ import annotations

import json
import sys

from jupyter_client.manager import start_new_kernel


def run(path: str, kernel: str = "rwd", timeout: int = 3600) -> int:
    nb = json.load(open(path, encoding="utf-8"))
    km, kc = start_new_kernel(kernel_name=kernel)
    errors = n = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            n += 1
            outs: list[dict] = []
            mid = kc.execute("".join(cell["source"]))
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["parent_header"].get("msg_id") != mid:
                    continue
                kind, c = msg["msg_type"], msg["content"]
                if kind == "status" and c["execution_state"] == "idle":
                    break
                if kind == "stream":
                    if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == c["name"]:
                        outs[-1]["text"] += c["text"]
                    else:
                        outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif kind in ("display_data", "execute_result"):
                    out = {"output_type": kind, "data": c["data"], "metadata": c["metadata"]}
                    if kind == "execute_result":
                        out["execution_count"] = n
                    outs.append(out)
                elif kind == "error":
                    errors += 1
                    outs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
                    print(f"ERROR in code cell {n}: {c['ename']}: {c['evalue']}")
            for out in outs:
                if out["output_type"] == "stream":
                    print(out["text"], end="")
                    out["text"] = out["text"].splitlines(True)
            cell["outputs"], cell["execution_count"] = outs, n
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
