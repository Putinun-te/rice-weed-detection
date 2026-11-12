"""Git clean filter: drop outputs and execution counts from a notebook (stdin -> stdout).

Notebook outputs can hold third-party images and unpublished numbers, which must not reach git.
With this filter the outputs stay in the working copy and only the code and text are committed.
Enabled by scripts/setup_env.sh (git config filter.nbstrip.clean) and .gitattributes.
"""
from __future__ import annotations

import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def main() -> int:
    nb = strip(json.loads(sys.stdin.buffer.read().decode("utf-8")))
    out = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(out.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
