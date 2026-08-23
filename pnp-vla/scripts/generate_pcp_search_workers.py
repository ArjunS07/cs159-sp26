"""Generate thin PCP-search Colab worker launchers."""
from __future__ import annotations

from pathlib import Path

from nb_common import bootstrap, code, md, notebook, write_notebook


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "workers"
SHARD_COUNT = 4
MANIFEST_ID = "pcps-c810651498933ba955c51560"


def worker_notebook(shard_index: int) -> dict:
    name = f"52_pcp_search_worker_{shard_index}.ipynb"
    return notebook([
        md(
            f"# PCP-search collection worker {shard_index}/{SHARD_COUNT}\n\n"
            "Thin GPU launcher. The frozen manifest and every collection/validation detail are "
            "owned by `pnp.pcp_search`."),
        code(bootstrap("sim", setup_env=True)),
        code(
            "from pnp.pcp_search.collection import run_pcp_search_worker\n\n"
            f"MANIFEST_ID = {MANIFEST_ID!r}\n"
            f"SHARD_COUNT = {SHARD_COUNT}\n"
            f"SHARD_INDEX = {shard_index}\n"
            "# Runtime throughput knob: independent same-task environments per GPU policy call.\n"
            "# Lower this only if this Colab GPU OOMs; it does not change rollout semantics.\n"
            "ROLLOUT_BATCH_SIZE = 8\n"
            "assert MANIFEST_ID.startswith(\"pcps-\"), \"Expected a frozen PCP-search manifest ID\"\n"
            "run_pcp_search_worker(\n"
            "    manifest_id=MANIFEST_ID, shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,\n"
            "    rollout_batch_size=ROLLOUT_BATCH_SIZE)"),
    ], name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for shard_index in range(SHARD_COUNT):
        path = OUT / f"52_pcp_search_worker_{shard_index}.ipynb"
        write_notebook(path, worker_notebook(shard_index))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
