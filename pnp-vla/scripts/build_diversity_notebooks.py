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
The shared split manifest is stored in Drive so both runs use exactly the same experiment.

This is a **diversity-signal pilot**, not a reproduction of published LIBERO performance. The
official OpenPI recipe trains raw pi0.5 for 30k steps with batch size 256 and a 10-action horizon;
this pilot uses 3k steps, batch size 16, and keeps the 50-action chunk / execute-10 interface used
by the PnP experiments. The training shim rebuilds the raw checkpoint's stale processor metadata
with the exact pinned LeRobot implementation; it does not alter the model weights or turn on an
extra relative-action transform."""),
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
SAVE_FREQ = 10000               # ignored while SAVE_CHECKPOINTS=False
SAVE_CHECKPOINTS = False        # no optimizer checkpoints; final inference model goes to Drive
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
CHECKPOINT_MIRROR_DIR = PERSISTENT_ROOT / f"checkpoint_m{MODEL_INDEX}"
FINAL_DRIVE_DIR = PERSISTENT_ROOT / f"final_model_m{MODEL_INDEX}"
print({"member": MODEL_INDEX, "model_repo": MODEL_REPOS[MODEL_INDEX],
       "output": str(OUTPUT_DIR), "checkpoint_mirror": str(CHECKPOINT_MIRROR_DIR),
       "final_drive_dir": str(FINAL_DRIVE_DIR), "full_finetune": FULL_FINETUNE,
       "save_checkpoints": SAVE_CHECKPOINTS})'''),
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
    md("""## 4. Launch training

Expected after the weights load: `Built fresh pinned-LeRobot processors with LIBERO camera
mapping.` This confirms that the obsolete processor metadata in `pi05_base` was replaced by the
pinned implementation before optimizer creation.

Training stays on fast local disk. After each complete save, the checkpoint is verified and
mirrored to Drive, then all local checkpoint copies and older Drive mirrors are removed. If a
fresh Colab runtime starts with `RESUME=True`, the newest complete Drive checkpoint is preferred;
`/content` is used only if Drive has no complete checkpoint. Partial saves are ignored. The
temporary checkpoint copy is removed after model, optimizer, and scheduler state are loaded.
Do not manually delete checkpoint folders while a save is in progress.

With the default `SAVE_CHECKPOINTS=False`, LeRobot writes no training checkpoints at all; the final
model and processors are exported directly to `FINAL_DRIVE_DIR` before the Hub upload is attempted.
This model-only export omits the large optimizer state, so it is suitable for evaluation but not
training resume. A disconnect before step 3000 still cannot be resumed from a newer step."""),
    code(r'''import subprocess, sys
from pathlib import Path

args = [sys.executable, "-u",
        str(Path(package_dir) / "scripts" / "train_pi05_bootstrap.py"),
        "--manifest", str(MANIFEST_PATH), "--member", str(MODEL_INDEX),
        "--output-dir", str(OUTPUT_DIR),
        "--checkpoint-mirror-dir", str(CHECKPOINT_MIRROR_DIR),
        "--final-drive-dir", str(FINAL_DRIVE_DIR),
        "--policy-repo-id", MODEL_REPOS[MODEL_INDEX],
        "--steps", str(STEPS), "--batch-size", str(BATCH_SIZE),
        "--save-freq", str(SAVE_FREQ)]
if not FULL_FINETUNE: args.append("--expert-only")
if COMPILE_MODEL: args.append("--compile-model")
if WANDB: args.append("--wandb")
if RESUME: args.append("--resume")
if not SAVE_CHECKPOINTS: args.append("--no-checkpoints")
print("starting member", MODEL_INDEX)
process = subprocess.Popen(
    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1)
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
if return_code:
    raise RuntimeError(f"Training exited with code {return_code}")'''),
    md("## 5. Record the immutable identifiers"),
    code(r'''print({"member": MODEL_INDEX, "model_repo": MODEL_REPOS[MODEL_INDEX],
       "manifest_hash": manifest["manifest_hash"], "source_model": manifest["source_model"],
       "steps": STEPS, "batch_size": BATCH_SIZE, "full_finetune": FULL_FINETUNE})
print("Before training member 1, preserve and reuse:", MANIFEST_PATH)'''),
], "17_train_diverse_pi05.ipynb")


TRAIN_V2 = notebook([
    md("""# 17 v2 - Two competent, bootstrapped pi0.5 LIBERO members

Run this notebook twice on an **A100/H100 80GB**, first with `MODEL_INDEX=0`, then with `1`.
Unlike the raw-base signal pilot, both members start from the same LIBERO-finetuned checkpoint.
They retain all 40 LIBERO tasks but use independent within-task episode bootstraps and seeds.

The target is a better competence/diversity tradeoff: 6,000 full-model updates at batch 32 and a
conservative `5e-6` learning rate. This is a new v2 experiment with separate Drive paths, manifest,
Hub repositories, and future Supabase labels; it does not overwrite v1.

Full LeRobot checkpoints remain disabled. Every 1,000 global updates, the trainer atomically
replaces one model-only Drive recovery bundle (weights + processors, no optimizer). If Colab
disconnects, restart the runtime and rerun the notebook for the same member: training warm-starts
from that bundle with a fresh optimizer. At completion, the final model is saved to Drive and
validated before Hugging Face upload is attempted."""),
    md("## 1. Install the exact pinned training stack"),
    code(bootstrap(extras="train", setup_env=False)),
    md("## 2. Configuration and isolated v2 paths"),
    code(r'''import os
from pathlib import Path
import torch
from google.colab import drive
from huggingface_hub import HfApi
from pnp.config import PI05_REPO_ID

drive.mount("/content/drive")

MODEL_INDEX = 0                 # change only this to 1 for the second member
TARGET_STEPS = 6000
BATCH_SIZE = 32                 # observed batch 16 used 40.4/80 GB; drop to 24 if this OOMs
LEARNING_RATE = 5e-6
SCHEDULER_WARMUP_STEPS = 500
SCHEDULER_DECAY_STEPS = 6000
SCHEDULER_DECAY_LR = 5e-7
RECOVERY_EVERY = 1000           # one rolling model-only bundle; never optimizer state
FULL_FINETUNE = True
COMPILE_MODEL = False
WANDB = False

assert MODEL_INDEX in (0, 1)
assert FULL_FINETUNE, "v2 is predeclared as a full-model fine-tune"
gpu = torch.cuda.get_device_properties(0)
assert gpu.total_memory / 2**30 >= 70, "Use an A100/H100 80GB runtime"

SOURCE_MODEL = PI05_REPO_ID     # lerobot/pi05_libero_finetuned
HF_USER = HfApi().whoami()["name"]
MODEL_REPOS = [f"{HF_USER}/pi05-libero-ft-bootstrap-m0-v2",
               f"{HF_USER}/pi05-libero-ft-bootstrap-m1-v2"]
PERSISTENT_ROOT = Path("/content/drive/MyDrive/pnp_diversity_v2")
MANIFEST_PATH = PERSISTENT_ROOT / "bootstrap_manifest_finetuned_v2.json"
OUTPUT_DIR = Path(f"/content/pi05_diversity_v2/train_m{MODEL_INDEX}")
RECOVERY_DIR = PERSISTENT_ROOT / f"recovery_m{MODEL_INDEX}"
FINAL_DRIVE_DIR = PERSISTENT_ROOT / f"final_model_m{MODEL_INDEX}"

print({"gpu": gpu.name, "gpu_gib": round(gpu.total_memory / 2**30, 1),
       "member": MODEL_INDEX, "source": SOURCE_MODEL,
       "model_repo": MODEL_REPOS[MODEL_INDEX], "output": str(OUTPUT_DIR),
       "recovery": str(RECOVERY_DIR), "final": str(FINAL_DRIVE_DIR),
       "steps": TARGET_STEPS, "batch": BATCH_SIZE, "lr": LEARNING_RATE})'''),
    md("""## 3. Build or load the shared v2 bootstrap manifest

Member 1 must load this exact file. The source checkpoint and dataset revisions are pinned inside
the manifest; a stale v1/raw-base manifest fails loudly instead of being reused."""),
    code(r'''from pnp.diversity import (bootstrap_manifest_summary,
    build_bootstrap_manifest_from_lerobot, load_bootstrap_manifest,
    save_bootstrap_manifest)

if MANIFEST_PATH.exists():
    manifest = load_bootstrap_manifest(MANIFEST_PATH)
else:
    manifest = build_bootstrap_manifest_from_lerobot(
        source_model=SOURCE_MODEL, seed=260)
    save_bootstrap_manifest(manifest, MANIFEST_PATH)

assert manifest["source_model"] == SOURCE_MODEL, manifest["source_model"]
assert manifest["n_tasks"] == 40 and manifest["n_source_episodes"] == 1693
display(bootstrap_manifest_summary(manifest))
print("manifest:", MANIFEST_PATH)
print("manifest hash:", manifest["manifest_hash"])
print("dataset revision:", manifest["dataset_revision"])
print("source model revision:", manifest["source_model_revision"])'''),
    md("""## 4. Download immutable inputs before training

This exposes download failures before GPU training starts. Both LeRobot and Hugging Face reuse the
same local Colab cache; never place the large cache on Drive."""),
    code(r'''from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_LEROBOT_HOME"] = str(HF_HOME)
snapshot_download(
    repo_id=manifest["dataset_repo_id"], repo_type="dataset",
    revision=manifest["dataset_revision"], max_workers=2)
snapshot_download(
    repo_id=manifest["source_model"],
    revision=manifest["source_model_revision"], max_workers=2)
print("Immutable dataset and source checkpoint are cached in:", HF_HOME)'''),
    md("""## 5. Verify Hub destination and launch

The first 20-50 updates are the memory preflight. Watch `nvidia-smi`; batch 32 should stay below
roughly 72 GiB. If it OOMs, restart the runtime and set `BATCH_SIZE=24`.

`--no-checkpoints` is unconditional. Recovery snapshots are model-only and keep exactly one Drive
copy. On a warm restart, the printed `global_start_step` is the saved global step and
`session_steps` is only the remaining work; optimizer/scheduler state starts fresh."""),
    code(r'''from google.colab import userdata

token = userdata.get("HF_TOKEN")
assert token, "Grant this notebook access to the HF_TOKEN Colab secret"
os.environ["HF_TOKEN"] = token
api = HfApi(token=token)
assert api.whoami()["name"] == HF_USER
api.create_repo(repo_id=MODEL_REPOS[MODEL_INDEX], repo_type="model", exist_ok=True)
print("Write access verified:", MODEL_REPOS[MODEL_INDEX])'''),
    code(r'''import subprocess, sys

args = [sys.executable, "-u",
        str(Path(package_dir) / "scripts" / "train_pi05_bootstrap.py"),
        "--manifest", str(MANIFEST_PATH), "--member", str(MODEL_INDEX),
        "--output-dir", str(OUTPUT_DIR),
        "--policy-repo-id", MODEL_REPOS[MODEL_INDEX],
        "--steps", str(TARGET_STEPS), "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--scheduler-warmup-steps", str(SCHEDULER_WARMUP_STEPS),
        "--scheduler-decay-steps", str(SCHEDULER_DECAY_STEPS),
        "--scheduler-decay-lr", str(SCHEDULER_DECAY_LR),
        "--save-freq", str(TARGET_STEPS + 1), "--no-checkpoints",
        "--model-only-recovery-dir", str(RECOVERY_DIR),
        "--model-only-recovery-freq", str(RECOVERY_EVERY),
        "--final-drive-dir", str(FINAL_DRIVE_DIR)]
if not FULL_FINETUNE: args.append("--expert-only")
if COMPILE_MODEL: args.append("--compile-model")
if WANDB: args.append("--wandb")

print("starting v2 member", MODEL_INDEX)
process = subprocess.Popen(
    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=0)
for character in iter(lambda: process.stdout.read(1), ""):
    print(character, end="", flush=True)
return_code = process.wait()
if return_code:
    raise RuntimeError(f"Training exited with code {return_code}")'''),
    md("## 6. Verify the durable final model"),
    code(r'''import json

metadata_path = FINAL_DRIVE_DIR / "final_export.json"
assert metadata_path.is_file(), f"missing final export: {metadata_path}"
final_metadata = json.loads(metadata_path.read_text())
assert final_metadata["member_index"] == MODEL_INDEX
assert final_metadata["training_completed_steps"] == TARGET_STEPS
assert final_metadata["manifest_hash"] == manifest["manifest_hash"]
assert list(FINAL_DRIVE_DIR.glob("*.safetensors")), "missing final model weights"
assert not RECOVERY_DIR.exists(), "completed final export should remove superseded recovery"

remote = api.model_info(MODEL_REPOS[MODEL_INDEX])
print({"member": MODEL_INDEX, "repo": MODEL_REPOS[MODEL_INDEX],
       "remote_revision": remote.sha, "manifest_hash": manifest["manifest_hash"],
       "source_model": manifest["source_model"], "source_revision": manifest["source_model_revision"],
       "steps": TARGET_STEPS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
       "final_drive_dir": str(FINAL_DRIVE_DIR)})
print("For member 1, change only MODEL_INDEX and reuse:", MANIFEST_PATH)'''),
], "17_train_diverse_pi05_v2.ipynb")


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
    write_notebook(ROOT / "notebooks" / "17_train_diverse_pi05_v2.ipynb", TRAIN_V2)
    for member_index in (0, 1):
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"18_diversity_signal_model_{member_index}.ipynb",
                       collection_notebook(member_index))
    write_notebook(ROOT / "notebooks" / "19_analyze_diversity_signal.ipynb", ANALYZE)


if __name__ == "__main__":
    main()
