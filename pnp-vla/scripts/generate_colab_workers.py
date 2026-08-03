"""Generate stable, thin Colab launchers; experiment logic stays in pnp.experiments."""
from __future__ import annotations

import json
from pathlib import Path

from nb_common import bootstrap


ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "workers"
SHARD_COUNT = 6

# Clone/install [sim] and run env setup in one fetched bootstrap cell.
BOOTSTRAP = bootstrap("sim", setup_env=True)


def cell(cell_type: str, source: str) -> dict:
    value = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def notebook(shard_index: int, *, benchmark: str = "libero") -> dict:
    if benchmark == "libero":
        filename = f"libero_worker_{shard_index}.ipynb"
        title = f"LIBERO collection worker {shard_index}/{SHARD_COUNT}"
        function = "run_libero_hybrid_worker"
    elif benchmark == "libero_pro":
        filename = f"libero_pro_worker_{shard_index}.ipynb"
        title = f"LIBERO-PRO canonical worker {shard_index}/{SHARD_COUNT}"
        function = "run_libero_pro_worker"
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    run = f'''from pnp.experiments import {function}

SHARD_COUNT = {SHARD_COUNT}
SHARD_INDEX = {shard_index}
{function}(shard_count=SHARD_COUNT, shard_index=SHARD_INDEX)
'''
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("markdown", f"# {title}\n\n"
                 "Stable launcher: all mutable experiment logic is pulled from `pnp.experiments`.\n"),
            cell("code", BOOTSTRAP),
            cell("code", run),
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for benchmark in ("libero", "libero_pro"):
        for shard_index in range(SHARD_COUNT):
            stem = "libero_worker" if benchmark == "libero" else "libero_pro_worker"
            path = OUT / f"{stem}_{shard_index}.ipynb"
            path.write_text(json.dumps(
                notebook(shard_index, benchmark=benchmark), indent=1) + "\n")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
