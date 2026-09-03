"""Generate the two thin Colab workers for the five-step diversity pilot."""
from __future__ import annotations

from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


SHARD_COUNT = 2


def worker_notebook(shard_index: int):
    return notebook([
        md(f"""# 60 - Five-step diversity PRO220 worker {shard_index}/{SHARD_COUNT}

This worker runs **110 frozen identities** and **three new rollouts per identity** (330 rollouts
total): five-step x1, five-step x3 lowest-U20, and five-step x3 select-then-refine. Every model
output is still a 50-action chunk; every arm executes exactly 10 actions before replanning.

Candidate selection uses K=5 uncertainty at zero-based five-step Euler indices `(2, 3)`, whose
flow times are `(0.6, 0.4)`. It ranks by U20 while retaining U10/U20/U50 and all three candidate
chunks. No absolute uncertainty threshold is used.

Worker 0 may first set `EPISODE_LIMIT=1` for a three-rollout sentinel. Restore it to `None` and
rerun the same worker; the completed rows are resume-safe and will be reused. Run workers 0 and 1
against the same `main` commit and shared v2 manifest."""),
        md("## 1. Setup a fresh GPU runtime"),
        code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and resumable collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from pnp.config import PI05_REPO_ID
from pnp.diversity import load_bootstrap_manifest
from pnp.five_step_diversity_experiment import (
    FIVE_STEP_DIVERSITY_EPISODE_INDICES, FIVE_STEP_DIVERSITY_EXPERIMENT,
    FIVE_STEP_DIVERSITY_PROBE_STEPS, FIVE_STEP_DIVERSITY_SHARD_COUNT,
    run_five_step_diversity_worker, validate_five_step_diversity_sentinel)

drive.mount("/content/drive")

EPISODE_INDICES = FIVE_STEP_DIVERSITY_EPISODE_INDICES  # fixed: (10, 11)
PROBE_STEPS = FIVE_STEP_DIVERSITY_PROBE_STEPS          # zero-based: (2, 3)
SHARD_COUNT = FIVE_STEP_DIVERSITY_SHARD_COUNT          # fixed: 2
SHARD_INDEX = {shard_index}
EPISODE_LIMIT = None  # worker-0 sentinel only: set 1 once, then restore None
EXPERIMENT = FIVE_STEP_DIVERSITY_EXPERIMENT
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]
SOURCE_MODEL_REVISION = manifest["source_model_revision"]
assert SOURCE_MODEL_REVISION, "v2 manifest is missing source_model_revision"
assert SHARD_COUNT == 2 and SHARD_INDEX in (0, 1)
assert PROBE_STEPS == (2, 3)

print({{"experiment": EXPERIMENT, "episode_indices": EPISODE_INDICES,
       "probe_steps_zero_based": PROBE_STEPS, "probe_flow_times": (0.6, 0.4),
       "num_inference_steps": 5, "num_queries": 3, "pnp_k": 5,
       "selection_horizon": 20, "n_action_steps": 10, "generated_chunk_size": 50,
       "identities_in_full_shard": 110, "rollouts_in_full_shard": 330,
       "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX,
       "episode_limit": EPISODE_LIMIT, "manifest_hash": manifest["manifest_hash"],
       "source_model_revision": SOURCE_MODEL_REVISION,
       "absolute_uncertainty_threshold": None}})

run_five_step_diversity_worker(
    episode_indices=EPISODE_INDICES, episode_limit=EPISODE_LIMIT,
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
    manifest_hash=manifest["manifest_hash"],
    source_model_revision=SOURCE_MODEL_REVISION,
    experiment=EXPERIMENT)'''),
        md("## 3. Validate one completed three-arm identity from this shard"),
        code('''validation = validate_five_step_diversity_sentinel(
    shard_index=SHARD_INDEX,
    source_id=f"{PI05_REPO_ID}@{SOURCE_MODEL_REVISION}",
    experiment=EXPERIMENT)
assert validation["status"] == "passed"
validation'''),
    ], f"60_five_step_diversity_pro220_worker_{shard_index}.ipynb")


def main():
    for shard_index in range(SHARD_COUNT):
        path = ROOT / "notebooks" / "workers" / (
            f"60_five_step_diversity_pro220_worker_{shard_index}.ipynb")
        write_notebook(path, worker_notebook(shard_index))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
