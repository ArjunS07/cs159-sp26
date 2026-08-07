"""Generate the two-model bootstrap training, PRO collection, and analysis notebooks."""
from __future__ import annotations

from nb_common import ROOT, bootstrap, code, md, notebook, write_notebook


TRAIN = notebook([
    md("""# 17 — Train one member of the two-model diversity experiment

Run this notebook twice on an **A100/H100 80GB**, first with `MODEL_INDEX=0`, then with `1`.
Both runs use one shared manifest: every LIBERO task is retained, while demonstrations are
independently sampled with replacement within task. This is a full-model fine-tune by default.

The start point is the agreed raw `lerobot/pi05_base`. Neither member inherits the common
LIBERO-finetuned solution; their LIBERO specialization comes only from their independently
bootstrapped demonstration multisets. Final weights are pushed to your Hugging Face account.
The shared split manifest is stored in Drive so both runs use exactly the same experiment."""),
    md("## 1. Install the exact pinned training stack"),
    code(bootstrap(extras="train", setup_env=False)),
    md("## 2. Configuration"),
    code(r'''from pathlib import Path
from huggingface_hub import HfApi
from google.colab import drive

drive.mount("/content/drive")

MODEL_INDEX = 0                 # rerun a separate copy with 1
STEPS = 3000                    # signal pilot; increase only after both models train cleanly
BATCH_SIZE = 16                 # safe starting point on an 80GB A100/H100
SAVE_FREQ = 500                 # resumable checkpoints within the current Colab runtime
FULL_FINETUNE = True            # planned experiment; False is explicit expert-only fallback
COMPILE_MODEL = False           # avoids a long first-step compile during the pilot
WANDB = False
RESUME = False

HF_USER = HfApi().whoami()["name"]
MODEL_REPOS = [f"{HF_USER}/pi05-base-to-libero-bootstrap-m0-v1",
               f"{HF_USER}/pi05-base-to-libero-bootstrap-m1-v1"]
PERSISTENT_ROOT = Path("/content/drive/MyDrive/pnp_diversity")
MANIFEST_PATH = PERSISTENT_ROOT / "bootstrap_manifest.json"
OUTPUT_DIR = Path(f"/content/pi05_diversity/train_m{MODEL_INDEX}")
print({"member": MODEL_INDEX, "model_repo": MODEL_REPOS[MODEL_INDEX],
       "output": str(OUTPUT_DIR), "full_finetune": FULL_FINETUNE})'''),
    md("""## 3. Build or load the shared episode-bootstrap manifest

The file in Drive contains both independently sampled members. Member 1 must load this exact file;
it must not generate a second manifest."""),
    code(r'''from pnp.diversity import (bootstrap_manifest_summary,
    build_bootstrap_manifest_from_lerobot, load_bootstrap_manifest,
    save_bootstrap_manifest)

if MANIFEST_PATH.exists():
    manifest = load_bootstrap_manifest(MANIFEST_PATH)
else:
    manifest = build_bootstrap_manifest_from_lerobot()
    save_bootstrap_manifest(manifest, MANIFEST_PATH)
assert manifest["source_model"] == "lerobot/pi05_base", manifest["source_model"]
display(bootstrap_manifest_summary(manifest))
print("manifest:", MANIFEST_PATH)
print("manifest hash:", manifest["manifest_hash"])
print("dataset revision:", manifest["dataset_revision"])
print("raw model revision:", manifest["source_model_revision"])
print("tasks:", manifest["n_tasks"], "source episodes:", manifest["n_source_episodes"])'''),
    md("## 4. Launch training"),
    code(r'''import subprocess, sys

args = [sys.executable, str(package_dir / "scripts" / "train_pi05_bootstrap.py"),
        "--manifest", str(MANIFEST_PATH), "--member", str(MODEL_INDEX),
        "--output-dir", str(OUTPUT_DIR),
        "--policy-repo-id", MODEL_REPOS[MODEL_INDEX],
        "--steps", str(STEPS), "--batch-size", str(BATCH_SIZE),
        "--save-freq", str(SAVE_FREQ)]
if not FULL_FINETUNE: args.append("--expert-only")
if COMPILE_MODEL: args.append("--compile-model")
if WANDB: args.append("--wandb")
if RESUME: args.append("--resume")
print("starting member", MODEL_INDEX)
subprocess.run(args, check=True)'''),
    md("## 5. Record the immutable identifiers"),
    code(r'''print({"member": MODEL_INDEX, "model_repo": MODEL_REPOS[MODEL_INDEX],
       "manifest_hash": manifest["manifest_hash"], "source_model": manifest["source_model"],
       "steps": STEPS, "batch_size": BATCH_SIZE, "full_finetune": FULL_FINETUNE})
print("Before training member 1, preserve and reuse:", MANIFEST_PATH)'''),
], "17_train_diverse_pi05.ipynb")


def collection_notebook(member_index: int):
    return notebook([
        md(f"""# 18 — Diversity-signal LIBERO-PRO worker: model {member_index}

Run model 0 and model 1 workers independently. They evaluate identical 13-suite PRO identities,
with identical rollout and perturbation seeds. The action is unchanged by the K=5 probe; the run
records uncertainty plus exact generated chunks for the first-decision diversity analysis."""),
        md("## 1. Setup a fresh GPU runtime"), code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from huggingface_hub import HfApi
from pnp.diversity import load_bootstrap_manifest, run_diversity_signal_worker

drive.mount("/content/drive")

MEMBER_INDEX = {member_index}
HF_USER = HfApi().whoami()["name"]
MODEL_REPO = f"{{HF_USER}}/pi05-base-to-libero-bootstrap-m{{MEMBER_INDEX}}-v1"
EPISODES_PER_TASK = 2          # 260 matched rollouts/model; raise to 5 after the signal pilot
SHARD_COUNT = 1
SHARD_INDEX = 0
MANIFEST_PATH = Path("/content/drive/MyDrive/pnp_diversity/bootstrap_manifest.json")
MANIFEST_HASH = load_bootstrap_manifest(MANIFEST_PATH)["manifest_hash"]

print({{"member": MEMBER_INDEX, "model_repo": MODEL_REPO,
       "episodes_per_task": EPISODES_PER_TASK}})
run_diversity_signal_worker(
    member_index=MEMBER_INDEX, model_repo_id=MODEL_REPO,
    episodes_per_task=EPISODES_PER_TASK,
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
    manifest_hash=MANIFEST_HASH)'''),
    ], f"18_diversity_signal_model_{member_index}.ipynb")


ANALYZE = notebook([
    md("""# 19 — Analyze the two-model diversity signal

This is a zero-simulation analysis of the two matched model experiments. The primary selector proxy
is whether lower first-chunk uncertainty selects the model that later succeeds. The two rollouts
use matched simulator identities and seeds, but run separately, so this is a signal test rather
than an online two-model policy. Whole-episode uncertainty is post-hoc only, and the oracle row
measures complementarity rather than an achievable policy."""),
    md("## 1. Setup"), code(bootstrap(extras="analysis", setup_env=False)),
    md("## 2. Fetch and validate the matched experiments"),
    code(r'''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from tqdm.auto import tqdm
from pnp.store import SupabaseStore
from pnp.diversity import (action_disagreement_summary, add_first_chunk_action_disagreement,
    analyze_diversity_signal, diversity_signal_figures, fetch_diversity_signal)

OUTPUT = Path("diversity_signal_outputs")
OUTPUT.mkdir(exist_ok=True)
store = SupabaseStore()
rollouts, steps = fetch_diversity_signal(store)
print({"rollouts": len(rollouts), "step rows": len(steps),
       "experiments": sorted(rollouts.experiment.unique())})
tables = analyze_diversity_signal(rollouts, steps)
display(tables["diversity_signal_overall"])
display(tables["diversity_signal_by_suite"])'''),
    md("""## 3. Exact first-decision action diversity

This downloads the small generated-chunk artifacts. Disable it only for a quick table check. The
comparison uses the first 10 actions at the matched initial state. Because the simulations are
separate, this measures model plus render variability; later chunks are not compared because the
model trajectories have diverged by then."""),
    code(r'''LOAD_GENERATED_CHUNKS = True
if LOAD_GENERATED_CHUNKS:
    paired = add_first_chunk_action_disagreement(
        store, tables["diversity_paired_episodes"], progress=tqdm)
    tables["diversity_paired_episodes"] = paired
    tables["diversity_action_disagreement"] = action_disagreement_summary(paired)
    display(tables["diversity_action_disagreement"])'''),
    md("## 4. Figures and saved tables"),
    code(r'''for name, frame in tables.items(): frame.to_csv(OUTPUT / f"{name}.csv", index=False)
paths = diversity_signal_figures(tables, OUTPUT / "figures")
for path in paths:
    print(path.name); display(Image(filename=str(path)))'''),
    md("## 5. Concise readout"),
    code(r'''row = tables["diversity_signal_overall"].iloc[0]
print("Complementarity / oracle opportunity")
print("  discordant outcomes: %.1f%% (%d/%d)" %
      (100*row.discordant_fraction, row.n_discordant, row.n_pairs))
print("  best single model SR: %.1f%%" % (100*row.best_member_sr))
print("  either-model oracle SR: %.1f%%  (gap %+0.1f pp)" %
      (100*row.oracle_either_success_sr,
       100*(row.oracle_either_success_sr-row.best_member_sr)))
print("First-observation uncertainty selector")
print("  selected SR: %.1f%%  (vs best member %+0.1f pp)" %
      (100*row.lower_first_chunk_u_sr,
       100*(row.lower_first_chunk_u_sr-row.best_member_sr)))
print("  accuracy on discordant pairs: %.1f%%; win AUC: %.3f" %
      (100*row.lower_first_chunk_u_accuracy_discordant,
       row.lower_first_chunk_u_win_auc))
print("Whole-episode selector is POST-HOC only: %.1f%%" %
      (100*row.lower_episode_u_sr_posthoc))'''),
], "19_analyze_diversity_signal.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "17_train_diverse_pi05.ipynb", TRAIN)
    for member_index in (0, 1):
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"18_diversity_signal_model_{member_index}.ipynb",
                       collection_notebook(member_index))
    write_notebook(ROOT / "notebooks" / "19_analyze_diversity_signal.ipynb", ANALYZE)


if __name__ == "__main__":
    main()
