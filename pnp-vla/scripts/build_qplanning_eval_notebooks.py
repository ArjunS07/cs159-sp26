"""Generate one PRO220 evaluation worker for each trained Q critic."""
from __future__ import annotations

from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


def worker_notebook(horizon: int):
    number = 66 if horizon == 10 else 67
    name = f"{number}_eval_qplanning_q{horizon}_pro220.ipynb"
    return notebook([
        md(f"""# {number} - Evaluate Q{horizon} planning on PRO220

One resume-safe worker evaluates the trained Q{horizon} critic on the frozen **220 identities**
(11 suites x 10 tasks x init-state indices 10 and 11). At every decision boundary it:

1. draws 64 same-observation PI0.5 candidates;
2. decodes each candidate with 3 Euler steps;
3. scores Q{horizon}, softmax-averages the best 16, and
4. executes the first 10 actions of that blended 50-action chunk.

There is no PnP refinement or uncertainty gate. Every 25 completed identities, the table compares
the planner with the exact episode-matched historical source baseline (10 decode steps, 10 executed
actions). Videos, observation frames, and generated chunks are off; compact trajectories and
planner diagnostics are retained.

Use a fresh GPU runtime. For a sentinel, set `EPISODE_LIMIT=1`; then restore `None` and rerun.
The completed sentinel is reused."""),
        md("## 1. Setup"),
        code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Checkpoint and cohort"),
        code(f'''from pathlib import Path
from google.colab import drive
from pnp.config import PI05_REPO_ID
from pnp.diversity import load_bootstrap_manifest

drive.mount("/content/drive")

HORIZON = {horizon}
EPISODE_LIMIT = None  # optional sentinel: 1
CANDIDATE_BATCH_SIZE = 8  # lower only if candidate generation runs out of GPU memory
CHECKPOINT_PATH = None  # optionally paste an exact checkpoint_step_008000.pt path

OUTPUT_ROOT = Path("/content/drive/MyDrive/pnp_qplanning_corrector")
if CHECKPOINT_PATH is None:
    candidates = sorted(
        OUTPUT_ROOT.glob(f"pcpcds-*/q{{HORIZON}}_full/checkpoint_step_008000.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Q{{HORIZON}} full checkpoint; found "
            f"{{len(candidates)}}: {{[str(path) for path in candidates]}}. "
            "Set CHECKPOINT_PATH explicitly.")
    CHECKPOINT_PATH = candidates[0]
else:
    CHECKPOINT_PATH = Path(CHECKPOINT_PATH)
print("checkpoint:", CHECKPOINT_PATH)

MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]
SOURCE_MODEL_REVISION = manifest["source_model_revision"]
assert SOURCE_MODEL_REVISION, "v2 manifest is missing source_model_revision"
print({{"horizon": HORIZON, "episode_limit": EPISODE_LIMIT,
       "candidate_batch_size": CANDIDATE_BATCH_SIZE,
       "manifest_hash": manifest["manifest_hash"],
       "source_model_revision": SOURCE_MODEL_REVISION}})'''),
        md("## 3. Run"),
        code('''from pnp.qplanning_eval_experiment import run_qplanning_eval_worker

report = run_qplanning_eval_worker(
    horizon=HORIZON,
    checkpoint_path=CHECKPOINT_PATH,
    episode_limit=EPISODE_LIMIT,
    candidate_batch_size=CANDIDATE_BATCH_SIZE,
    manifest_hash=manifest["manifest_hash"],
    source_model_revision=SOURCE_MODEL_REVISION)
report'''),
        md("## 4. Persisted sentinel audit"),
        code('''from pnp.qplanning_eval_experiment import validate_qplanning_eval_sentinel

validate_qplanning_eval_sentinel(
    horizon=HORIZON,
    checkpoint_id=report["checkpoint_id"],
    experiment=report["experiment"])'''),
    ], name)


def main():
    for horizon in (10, 50):
        number = 66 if horizon == 10 else 67
        path = ROOT / "notebooks" / "workers" / (
            f"{number}_eval_qplanning_q{horizon}_pro220.ipynb")
        write_notebook(path, worker_notebook(horizon))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
