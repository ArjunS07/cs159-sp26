"""Generate stable, thin Colab launchers; experiment logic stays in pnp.experiments."""
from __future__ import annotations

import json
from pathlib import Path

from nb_common import bootstrap


ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "workers"
SHARD_COUNT = 6
ROLLOUT_BATCH_SIZE = 2

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
    elif benchmark == "libero_pro_expanded":
        filename = f"libero_pro16_worker_{shard_index}.ipynb"
        title = f"LIBERO-PRO expanded 16-suite worker {shard_index}/{SHARD_COUNT}"
        function = "run_libero_pro_expanded_worker"
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    run = f'''from pnp.experiments import {function}

SHARD_COUNT = {SHARD_COUNT}
SHARD_INDEX = {shard_index}
ROLLOUT_BATCH_SIZE = {ROLLOUT_BATCH_SIZE}
{function}(shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
           rollout_batch_size=ROLLOUT_BATCH_SIZE)
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


STEMS = {
    "libero": "libero_worker",
    "libero_pro": "libero_pro_worker",
    "libero_pro_expanded": "libero_pro16_worker",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for benchmark, stem in STEMS.items():
        for shard_index in range(SHARD_COUNT):
            path = OUT / f"{stem}_{shard_index}.ipynb"
            path.write_text(json.dumps(
                notebook(shard_index, benchmark=benchmark), indent=1) + "\n")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
