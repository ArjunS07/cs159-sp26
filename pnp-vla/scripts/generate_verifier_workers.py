"""Generate sharded copies of the canonical verifier-pair collection notebook."""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEFAULT_SOURCE = ROOT / "notebooks" / "04_collect_verifier_pairs.ipynb"
DEFAULT_OUTPUT = ROOT / "notebooks" / "workers"


def _patch_worker(master: dict, shard_count: int, shard_index: int) -> dict:
    worker = copy.deepcopy(master)
    patched = 0
    for cell in worker.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "SHARD_COUNT =" not in source or "SHARD_INDEX =" not in source:
            continue
        source, count = re.subn(
            r"(?m)^SHARD_COUNT\s*=\s*\d+.*$",
            f"SHARD_COUNT = {shard_count}   # Generated worker count.",
            source,
        )
        if count != 1:
            raise RuntimeError(f"expected one SHARD_COUNT assignment, found {count}")
        source, count = re.subn(
            r"(?m)^SHARD_INDEX\s*=\s*\d+.*$",
            f"SHARD_INDEX = {shard_index}   # Generated worker index.",
            source,
        )
        if count != 1:
            raise RuntimeError(f"expected one SHARD_INDEX assignment, found {count}")
        cell["source"] = source.splitlines(True)
        patched += 1

    if patched != 1:
        raise RuntimeError(f"expected one shard configuration cell, patched {patched}")

    filename = f"04_verifier_pairs_worker_{shard_index}.ipynb"
    worker.setdefault("metadata", {}).setdefault("colab", {})["name"] = filename
    for cell in worker.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    worker["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Generated collection worker {shard_index}/{shard_count}\n",
            "\n",
            "Generated from `04_collect_verifier_pairs.ipynb`; edit the canonical notebook, "
            "not this copy.\n",
        ],
    })
    return worker


def generate_workers(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT,
                     shard_count: int = 3) -> list[Path]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    master = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index in range(shard_count):
        path = output / f"04_verifier_pairs_worker_{shard_index}.ipynb"
        path.write_text(json.dumps(
            _patch_worker(master, shard_count, shard_index), indent=1) + "\n")
        paths.append(path)
        print(path.relative_to(ROOT))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    generate_workers(args.source, args.output, args.shards)


if __name__ == "__main__":
    main()
