"""Generate two thin Colab workers for the single-query 5/3-step follow-up."""
from __future__ import annotations

from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


def worker_notebook(shard_index):
    name = f"62_coarse_single_refinement_pro220_worker_{shard_index}.ipynb"
    return notebook([
        md(f"""# 62 - Single-query coarse refinement PRO220 worker {shard_index}/2

Same frozen 220-identity development pilot and two-worker split as notebooks 60.
Each worker runs **110 identities x three arms = 330 new rollouts**:

| Arm | Integration steps | Zero-based PnP indices | Flow times | Refinement |
| --- | ---: | --- | --- | --- |
| 5-step x1 + refine | 5 | `(2, 3)` | 0.6, 0.4 | refine-last, K=5 |
| 3-step x1 + refine | 3 | `(2,)` | 1/3 | refine-last, K=5 |
| 3-step x1 | 3 | `(2,)` | 1/3 | none; measurement only |

Every output remains **50 actions**, with **10 executed before replanning**. These are single
refined solves, not candidate selection or a preliminary unrefined query followed by another
solve. No gradient steering or uncertainty threshold. Generated chunks, U10/U20/U50 profiles,
contraction diagnostics, inference time, and velocity-field counts are retained through the
ordinary rollout logging path. Refinement uses the established episode-seeded perturbation
stream; initial chunk-noise seeds match ordinary single-query policy seeds.

**Historical printouts:** at 25/50/75/100 completed three-arm identities and at the end; normally
75 rollout ticks per table, not 25. Every table shows suite and overall SR with counts for all
three new arms, stock 10-step, and all three previous five-step arms, restricted to exactly the
same completed identities. Historical rows are verified and reused, never rerun. Failed/error
rollouts do not count as completed identities and are retried on resume.

For a small GPU sentinel, set `EPISODE_LIMIT=1`, run, then restore `None` and rerun. Completed
sentinel rollouts are reused. Run both workers against the same pushed `main` revision."""),
        md("## 1. Setup a fresh GPU runtime"),
        code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and resumable collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from pnp.config import PI05_REPO_ID
from pnp.diversity import load_bootstrap_manifest
from pnp.coarse_refinement_experiment import (
    COARSE_REFINEMENT_EXPERIMENT, COARSE_REFINEMENT_SHARD_COUNT,
    run_coarse_refinement_worker)

drive.mount("/content/drive")

SHARD_COUNT = COARSE_REFINEMENT_SHARD_COUNT  # fixed: 2
SHARD_INDEX = {shard_index}
EPISODE_LIMIT = None  # optional sentinel: set 1, then restore None and resume
EXPERIMENT = COARSE_REFINEMENT_EXPERIMENT
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]
SOURCE_MODEL_REVISION = manifest["source_model_revision"]
assert SOURCE_MODEL_REVISION, "v2 manifest is missing source_model_revision"
assert SHARD_COUNT == 2 and SHARD_INDEX in (0, 1)

run_coarse_refinement_worker(
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX, episode_limit=EPISODE_LIMIT,
    manifest_hash=manifest["manifest_hash"], source_model_revision=SOURCE_MODEL_REVISION,
    experiment=EXPERIMENT)'''),
    ], name)


def main():
    for shard_index in range(2):
        name = f"62_coarse_single_refinement_pro220_worker_{shard_index}.ipynb"
        path = ROOT / "notebooks" / "workers" / name
        write_notebook(path, worker_notebook(shard_index))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
