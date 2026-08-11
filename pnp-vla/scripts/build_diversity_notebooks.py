"""Generate the two-model bootstrap training, PRO collection, and analysis notebooks."""
from __future__ import annotations

import copy

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


def collection_v2_notebook(member_index: int):
    return notebook([
        md(f"""# 18 v2 — Diversity-signal LIBERO-PRO worker: model {member_index}

This evaluates one LIBERO-finetuned/bootstrap v2 member on the same deterministic 65-identity
shard used for the raw-base v1 pilot: 13 suites and five identities per suite. Run both v2 model
workers before analysis. Supabase labels are isolated under `pi05-diversity-signal-v2-*`."""),
        md("## 1. Setup a fresh GPU runtime"), code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from huggingface_hub import HfApi
from pnp.config import PI05_REPO_ID
from pnp.diversity import (DIVERSITY_V2_EXPERIMENT_PREFIX,
    load_bootstrap_manifest, run_diversity_signal_worker)

drive.mount("/content/drive")

MEMBER_INDEX = {member_index}
HF_USER = HfApi().whoami()["name"]
MODEL_REPO = f"{{HF_USER}}/pi05-libero-ft-bootstrap-m{{MEMBER_INDEX}}-v2"
EPISODES_PER_TASK = 2
SHARD_COUNT = 4
SHARD_INDEX = 0
EXPERIMENT_PREFIX = DIVERSITY_V2_EXPERIMENT_PREFIX
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]

print({{"member": MEMBER_INDEX, "model_repo": MODEL_REPO,
       "experiment_prefix": EXPERIMENT_PREFIX,
       "episodes_per_task": EPISODES_PER_TASK,
       "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX,
       "manifest_hash": manifest["manifest_hash"]}})
run_diversity_signal_worker(
    member_index=MEMBER_INDEX, model_repo_id=MODEL_REPO,
    episodes_per_task=EPISODES_PER_TASK,
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
    manifest_hash=manifest["manifest_hash"],
    experiment_prefix=EXPERIMENT_PREFIX)'''),
    ], f"18_diversity_signal_v2_model_{member_index}.ipynb")


def selective_refinement_v2_notebook(member_index: int):
    return notebook([
        md(f"""# 20 v2 - Selective-refinement LIBERO-PRO worker: model {member_index}

This collects the 10-episodes-per-task subset of the 13-suite cohort: 1,300 identities per arm.
It uses the established K=5, Euler-steps `(3,4)`, refine-last configuration and requests baseline and
refinement under the existing `pi05-diversity-signal-v2-m{member_index}` experiment. Supabase
rollout IDs are behavior-derived, so the existing 130 baselines and any completed refinements are
reused; only missing rows execute.

The model repository and immutable revision are read from the existing baseline run metadata.
This fails rather than silently evaluating a newer Hub upload. Run `SHARD_INDEX=0,1,2,3`; each
completed shard is resumable. Every fifty new rollouts, the worker prints the current baseline SR,
current refinement SR, and historical reference SR by suite."""),
        md("## 1. Setup a fresh GPU runtime"), code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and resumable collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from pnp.config import PI05_REPO_ID
from pnp.diversity import (DIVERSITY_V2_EXPERIMENT_PREFIX,
    load_bootstrap_manifest, run_diversity_refinement_worker)

drive.mount("/content/drive")

MEMBER_INDEX = {member_index}
EPISODES_PER_TASK = 10         # 13 suites x 10 tasks x 10 episodes = 1,300 identities/arm
SHARD_COUNT = 4
SHARD_INDEX = 0                # run 0, 1, 2, 3 for this member
EXPERIMENT_PREFIX = DIVERSITY_V2_EXPERIMENT_PREFIX
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]

print({{"member": MEMBER_INDEX, "experiment_prefix": EXPERIMENT_PREFIX,
       "episodes_per_task": EPISODES_PER_TASK,
       "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX,
       "manifest_hash": manifest["manifest_hash"]}})
run_diversity_refinement_worker(
    member_index=MEMBER_INDEX,
    episodes_per_task=EPISODES_PER_TASK,
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
    manifest_hash=manifest["manifest_hash"],
    experiment_prefix=EXPERIMENT_PREFIX)'''),
    ], f"20_diversity_selective_refinement_v2_model_{member_index}.ipynb")


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


ANALYZE_V2 = copy.deepcopy(ANALYZE)
ANALYZE_V2["metadata"]["colab"]["name"] = "19_analyze_diversity_signal_v2.ipynb"
for cell in ANALYZE_V2["cells"]:
    source = cell.get("source", "")
    source = source.replace(
        "# 19 — Analyze the two-model diversity signal",
        "# 19 v2 — Analyze the fine-tuned two-model diversity signal")
    source = source.replace(
        "measures complementarity rather than an achievable policy.",
        "measures complementarity rather than an achievable policy.\n\n"
        "This notebook fetches only `pi05-diversity-signal-v2-m0/m1`; v1 rows are excluded.")
    source = source.replace(
        'OUTPUT = Path("diversity_signal_outputs")',
        'EXPERIMENT_PREFIX = "pi05-diversity-signal-v2"\n'
        'OUTPUT = Path("diversity_signal_v2_outputs")')
    source = source.replace(
        "rollouts, steps = fetch_diversity_signal(store)",
        "rollouts, steps = fetch_diversity_signal(\n"
        "    store, experiment_prefix=EXPERIMENT_PREFIX)\n"
        "expected = [f\"{EXPERIMENT_PREFIX}-m0\", f\"{EXPERIMENT_PREFIX}-m1\"]\n"
        "assert sorted(rollouts.experiment.unique()) == expected")
    cell["source"] = source


ANALYZE_SELECTIVE_REFINEMENT_V2 = notebook([
    md("""# 21 - Analyze two-model selective refinement (v2)

This notebook combines the existing uncertainty-only arms with the matched K=5 `(3,4)`
refine-last arms for both competent v2 models. The primary predeclared policy uses first-chunk
uncertainty: select the lower-U model, refine it for the episode only when U is at least `0.03`.

Two-model analyses use identities completed in all four member/method arms. The source + model 1
analysis is built separately, so it uses all 1,300 model-1 identities after model-1 shards 0-3
finish even if model-0 shards 2-3 have not been run.

Prefix, individual-chunk, and full-episode summaries are also analyzed. Only the first-chunk rule
is directly deployable from these independently simulated trajectories. Later scores are useful
post-hoc evidence about predictiveness and headroom, but they are not presented as an online
policy. For `prefix_N`, trajectories with fewer than N chunks use their whole-episode uncertainty,
so every prefix sweep retains the entire matched cohort in its SR denominator. The best in-sample
window is exploratory and is evaluated on the same matched cohort used to choose it."""),
    md("## 1. Setup"), code(bootstrap(extras="analysis", setup_env=False)),
    md("## 2. Fetch and validate the exact four-arm cohort"),
    code(r'''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from pnp.config import Method
from pnp.store import SupabaseStore
from pnp.diversity import (DIVERSITY_FIXED_REFINEMENT_THRESHOLD,
    DIVERSITY_PAIR_KEYS, DIVERSITY_V2_EXPERIMENT_PREFIX,
    analyze_diversity_selective_refinement,
    diversity_selective_refinement_figures, fetch_diversity_selective_refinement)

EXPERIMENT_PREFIX = DIVERSITY_V2_EXPERIMENT_PREFIX
FIXED_THRESHOLD = DIVERSITY_FIXED_REFINEMENT_THRESHOLD  # predeclared 0.03
REQUIRE_FULL_COHORT = False      # True only after shards 0,1,2,3 finish for both models
EXPECTED_FULL_M1_IDENTITIES = 1300
OUTPUT = Path("diversity_selective_refinement_v2_outputs")
OUTPUT.mkdir(exist_ok=True)

store = SupabaseStore()
all_rollouts, all_steps = fetch_diversity_selective_refinement(
    store, experiment_prefix=EXPERIMENT_PREFIX)
print("Available completed rows before the shared four-arm intersection")
display(all_rollouts.groupby(["member_index", "method"]).size().rename("n").reset_index())
arm_counts = (all_rollouts.drop_duplicates(DIVERSITY_PAIR_KEYS + ["member_index", "method"])
              .groupby(DIVERSITY_PAIR_KEYS).size().rename("n_arms").reset_index())
complete_keys = arm_counts[arm_counts.n_arms == 4][DIVERSITY_PAIR_KEYS]
rollouts = all_rollouts.merge(complete_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
steps = all_steps[all_steps.rollout_id.isin(rollouts.rollout_id)].copy()
counts = (rollouts.groupby(["member_index", "method"]).size()
          .rename("n").reset_index())
display(counts)
expected_methods = {Method.UNCERTAINTY, Method.REFINEMENT}
assert set(rollouts.method) == expected_methods
assert counts.n.nunique() == 1, "Four arms are not balanced after intersection"
N_PER_ARM = int(counts.n.iloc[0])
assert rollouts.suite.nunique() == 13
if REQUIRE_FULL_COHORT:
    assert N_PER_ARM == 1300, f"Full cohort requires 1,300/arm, found {N_PER_ARM}"
print({"analysis_mode": "FULL" if REQUIRE_FULL_COHORT else "PARTIAL PREVIEW",
       "identities_per_arm": N_PER_ARM,
       "ignored_incomplete_identity_rows": int((arm_counts.n_arms < 4).sum()),
       "rollouts": len(rollouts), "step_rows": len(steps),
       "experiments": sorted(rollouts.experiment.unique()),
       "fixed_threshold": FIXED_THRESHOLD})'''),
    md("## 3. Per-model baseline, refinement, AUC, and windows"),
    code(r'''tables = analyze_diversity_selective_refinement(
    rollouts, steps, fixed_threshold=FIXED_THRESHOLD)
horizons = ["first_chunk", "prefix_2_chunks", "prefix_4_chunks",
            "prefix_8_chunks", "full_episode"]
member_overall = tables["member_refinement_overall"]
member_columns = ["member_index", "score_name", "n_pairs", "baseline_sr",
                  "refinement_sr", "delta_pp", "delta_ci_low_pp", "delta_ci_high_pp",
                  "F_to_S", "S_to_F", "failure_auc"]
for member_index in (0, 1):
    print(f"Model {member_index}: overall SR, matched refinement delta, and failure AUC")
    display(member_overall[(member_overall.member_index == member_index) &
                           member_overall.score_name.isin(horizons)][member_columns])
    print(f"Model {member_index}: top exploratory uncertainty windows")
    member_top = tables["member_refinement_top_windows"]
    display(member_top[(member_top.member_index == member_index) &
                       member_top.score_name.isin(horizons) & (member_top["rank"] <= 5)][[
        "score_name", "rank", "lower", "upper", "n_refined", "selective_sr",
        "delta_pp", "selected_F_to_S", "selected_S_to_F"]])'''),
    md("""## 4. Match the shared source checkpoint on the same episodes

The v2 members started from `lerobot/pi05_libero_finetuned`. This loads that checkpoint's previous
K=5 `(3,4)` expanded-PRO rollouts and restricts them to the exact four-arm identity intersection.
Source baseline is the direct competence comparison; source refinement is included for context."""),
    code(r'''from pnp.config import PI05_REPO_ID
from pnp.experiments import PRO_EXPANDED_EXPERIMENT
from pnp.diversity import analyze_checkpoint_refinement

source_runs = pd.DataFrame(store.fetch_all(
    "experiment_runs", "run_id,model_repo_id,model_revision",
    configure=lambda query: query.eq("experiment", PRO_EXPANDED_EXPERIMENT),
    order_by=("run_id",)))
display(source_runs[["model_repo_id", "model_revision"]].drop_duplicates())
assert set(source_runs.model_repo_id.dropna()) == {PI05_REPO_ID}

source_rows = pd.DataFrame(store.fetch_all(
    "rollouts", "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,status,success,"
    "u_mean_episode,pnp_k,pnp_step_indices,refine_average",
    configure=lambda query: query.eq("experiment", PRO_EXPANDED_EXPERIMENT),
    order_by=("rollout_id",)))
source_rows = source_rows[
    source_rows.status.eq("completed") & source_rows.pnp_k.eq(5) &
    source_rows.pnp_step_indices.apply(lambda value: tuple(value or []) == (3, 4))]
source_observed_rows = source_rows[source_rows.method.eq(Method.UNCERTAINTY)].copy()
source_refined_rows = source_rows[
    source_rows.method.eq(Method.REFINEMENT) &
    ~source_rows.refine_average.fillna(False).astype(bool)].copy()
source_observed = source_observed_rows[
    DIVERSITY_PAIR_KEYS + ["success"]].rename(columns={"success": "source_baseline_success"})
source_refined = source_refined_rows[
    DIVERSITY_PAIR_KEYS + ["success"]].rename(columns={"success": "source_refinement_success"})
assert not source_observed.duplicated(DIVERSITY_PAIR_KEYS).any()
assert not source_refined.duplicated(DIVERSITY_PAIR_KEYS).any()

paired = tables["selective_refinement_paired_episodes"]
comparison = paired[DIVERSITY_PAIR_KEYS + ["success_observed_m0", "success_observed_m1"]].merge(
    source_observed, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
assert len(comparison), "No exact source-checkpoint episode matches were found"
comparison = comparison.merge(
    source_refined, on=DIVERSITY_PAIR_KEYS, how="left", validate="one_to_one")
print(f"Source checkpoint matched {len(comparison)}/{len(paired)} current identities exactly")
for column in ["success_observed_m0", "success_observed_m1", "source_baseline_success"]:
    comparison[column] = comparison[column].astype(bool)
comparison["source_refinement_success"] = comparison.source_refinement_success.astype("boolean")
source_overall = pd.DataFrame([{
    "n": len(comparison),
    "source_baseline_sr": comparison.source_baseline_success.mean(),
    "source_refinement_n": comparison.source_refinement_success.notna().sum(),
    "source_refinement_sr": comparison.source_refinement_success.mean(),
    "model0_baseline_sr": comparison.success_observed_m0.mean(),
    "model1_baseline_sr": comparison.success_observed_m1.mean(),
    "model0_delta_vs_source_pp": 100 * (
        comparison.success_observed_m0.mean() - comparison.source_baseline_success.mean()),
    "model1_delta_vs_source_pp": 100 * (
        comparison.success_observed_m1.mean() - comparison.source_baseline_success.mean()),
}])
source_by_suite = pd.DataFrame([{
    "suite": suite, "n": len(group),
    "source_baseline_sr": group.source_baseline_success.mean(),
    "source_refinement_n": group.source_refinement_success.notna().sum(),
    "source_refinement_sr": group.source_refinement_success.mean(),
    "model0_baseline_sr": group.success_observed_m0.mean(),
    "model1_baseline_sr": group.success_observed_m1.mean(),
} for suite, group in comparison.groupby("suite", sort=True)])
tables["source_checkpoint_matched_episodes"] = comparison
tables["source_checkpoint_overall"] = source_overall
tables["source_checkpoint_by_suite"] = source_by_suite
print("Matched shared-source checkpoint comparison")
display(source_overall)
display(source_by_suite)

# Run both window sweeps on the exact identities that have source baseline + refinement and all
# four current member/method arms. If any source refinement is missing, restrict the aggregation
# analysis too, so the source-versus-aggregation graph always uses identical episodes.
matched_keys = comparison[comparison.source_refinement_success.notna()][DIVERSITY_PAIR_KEYS]
assert len(matched_keys), "No exact episodes contain both source baseline and source refinement"
if len(matched_keys) != len(paired):
    matched_rollouts = rollouts.merge(matched_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
    matched_steps = steps[steps.rollout_id.isin(matched_rollouts.rollout_id)].copy()
    tables = analyze_diversity_selective_refinement(
        matched_rollouts, matched_steps, fixed_threshold=FIXED_THRESHOLD)
    paired = tables["selective_refinement_paired_episodes"]
    print(f"Restricted aggregation window analysis to {len(paired)} exact source matches")
source_analysis_rollouts = pd.concat(
    [source_observed_rows, source_refined_rows], ignore_index=True).merge(
        matched_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")

def fetch_pnp_steps(rollout_ids):
    rows = []
    rollout_ids = [str(value) for value in rollout_ids]
    for start in range(0, len(rollout_ids), 100):
        batch = rollout_ids[start:start + 100]
        rows.extend(store.fetch_all(
            "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
            configure=lambda query, ids=batch: query.in_("rollout_id", ids),
            order_by=("rollout_id",)))
    return pd.DataFrame(rows)

source_observed_ids = source_analysis_rollouts[
    source_analysis_rollouts.method.eq(Method.UNCERTAINTY)].rollout_id.astype(str).tolist()
source_steps = fetch_pnp_steps(source_observed_ids)
source_analysis = analyze_checkpoint_refinement(
    source_analysis_rollouts, source_steps, checkpoint_name="source_checkpoint")
for name, frame in source_analysis.items():
    tables[name.replace("member_refinement", "source_checkpoint_refinement")] = frame

# Model 1 shards 0-3 are complete even when model 0 has only shards 0-1. Build an independent
# source/model-1 cohort instead of silently discarding model-1's second half in the four-arm join.
m1_rows = all_rollouts[
    all_rollouts.member_index.eq(1) & all_rollouts.status.eq("completed") &
    all_rollouts.method.isin(expected_methods)].copy()
assert not m1_rows.duplicated(DIVERSITY_PAIR_KEYS + ["method"]).any(), (
    "Model 1 contains duplicate method/episode identities")
m1_method_counts = (m1_rows.groupby(DIVERSITY_PAIR_KEYS).method.nunique()
                    .rename("n_methods").reset_index())
m1_complete_keys = m1_method_counts[m1_method_counts.n_methods.eq(2)][DIVERSITY_PAIR_KEYS]
source_complete_keys = source_observed[DIVERSITY_PAIR_KEYS].merge(
    source_refined[DIVERSITY_PAIR_KEYS], on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
source_m1_keys = m1_complete_keys.merge(
    source_complete_keys, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
assert len(source_m1_keys) == EXPECTED_FULL_M1_IDENTITIES, (
    f"Expected {EXPECTED_FULL_M1_IDENTITIES} exact source/model-1 identities after workers 0-3; "
    f"found {len(source_m1_keys)}")

m1_full_rollouts = m1_rows.merge(
    source_m1_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
m1_full_steps = all_steps[all_steps.rollout_id.isin(m1_full_rollouts.rollout_id)].copy()
m1_full_analysis = analyze_checkpoint_refinement(
    m1_full_rollouts, m1_full_steps, checkpoint_name=1)
source_m1_rollouts = pd.concat(
    [source_observed_rows, source_refined_rows], ignore_index=True).merge(
        source_m1_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
source_m1_observed_ids = source_m1_rollouts[
    source_m1_rollouts.method.eq(Method.UNCERTAINTY)].rollout_id.astype(str).tolist()
source_m1_steps = fetch_pnp_steps(source_m1_observed_ids)
source_m1_analysis = analyze_checkpoint_refinement(
    source_m1_rollouts, source_m1_steps, checkpoint_name="source_checkpoint_m1_cohort")
for name, frame in m1_full_analysis.items():
    tables[name.replace("member_refinement", "model1_full_refinement")] = frame
for name, frame in source_m1_analysis.items():
    tables[name.replace("member_refinement", "source_model1_cohort_refinement")] = frame
print({"source_plus_model1_exact_identities": len(source_m1_keys),
       "shared_source_model0_model1_identities": len(matched_keys)})

source_horizons = source_analysis["member_refinement_overall"]
print("Shared source checkpoint: matched refinement delta and failure AUC")
display(source_horizons[source_horizons.score_name.isin(horizons)][[
    "score_name", "n_pairs", "baseline_sr", "refinement_sr", "delta_pp",
    "delta_ci_low_pp", "delta_ci_high_pp", "F_to_S", "S_to_F", "failure_auc"]])
print("Shared source checkpoint: top exploratory uncertainty windows")
source_top = source_analysis["member_refinement_top_windows"]
display(source_top[source_top.score_name.isin(horizons) & (source_top["rank"] <= 5)][[
    "score_name", "rank", "lower", "upper", "n_refined", "selective_sr",
    "delta_pp", "selected_F_to_S", "selected_S_to_F"]])

# Main comparison: independently optimize a window for the source checkpoint and for the
# two-model lower-U aggregation, then compare whole-cohort SR gains on the same identities.
aggregate_best = tables["selective_refinement_best_windows"]
aggregate_overall = tables["selective_refinement_overall"]
source_best = source_analysis["member_refinement_top_windows"]
source_overall_windows = source_analysis["member_refinement_overall"]
window_comparison = (
    source_best[(source_best["rank"] == 1) & source_best.score_name.isin(horizons)][[
        "score_name", "lower", "upper", "n_refined", "selective_sr", "delta_pp"]]
    .rename(columns={
        "lower": "source_lower", "upper": "source_upper",
        "n_refined": "source_n_refined", "selective_sr": "source_window_sr",
        "delta_pp": "source_window_delta_pp"})
    .merge(source_overall_windows[source_overall_windows.score_name.isin(horizons)][[
        "score_name", "baseline_sr"]].rename(columns={"baseline_sr": "source_baseline_sr"}),
        on="score_name", validate="one_to_one")
    .merge(aggregate_best[aggregate_best.score_name.isin(horizons)][[
        "score_name", "lower", "upper", "n_refined", "selective_sr", "delta_pp",
        "delta_vs_best_fixed_pp"]].rename(columns={
            "lower": "aggregate_lower", "upper": "aggregate_upper",
            "n_refined": "aggregate_n_refined", "selective_sr": "aggregate_window_sr",
            "delta_pp": "aggregate_window_delta_pp",
            "delta_vs_best_fixed_pp": "aggregate_window_vs_best_member_pp"}),
        on="score_name", validate="one_to_one")
    .merge(aggregate_overall[aggregate_overall.score_name.isin(horizons)][[
        "score_name", "lower_u_baseline_sr", "best_fixed_member_sr"]],
        on="score_name", validate="one_to_one"))
window_comparison["aggregate_delta_advantage_pp"] = (
    window_comparison.aggregate_window_delta_pp - window_comparison.source_window_delta_pp)
window_comparison["aggregate_window_vs_source_window_pp"] = 100 * (
    window_comparison.aggregate_window_sr - window_comparison.source_window_sr)
window_comparison["score_name"] = pd.Categorical(
    window_comparison.score_name, categories=horizons, ordered=True)
window_comparison = window_comparison.sort_values("score_name")
tables["source_vs_aggregate_best_windows"] = window_comparison'''),
    md("""## 5. Main result: source model versus two-model aggregation

For each uncertainty horizon, both systems receive their own best lower/upper uncertainty window.
Every SR and delta uses the same matched episode cohort. `aggregate_delta_advantage_pp` is the key
column: positive means the two-model aggregation gains more SR from its window than the shared
source checkpoint gains from its own window."""),
    code(r'''main_columns = [
    "score_name",
    "source_baseline_sr", "source_lower", "source_upper", "source_n_refined",
    "source_window_sr", "source_window_delta_pp",
    "lower_u_baseline_sr", "aggregate_lower", "aggregate_upper", "aggregate_n_refined",
    "aggregate_window_sr", "aggregate_window_delta_pp",
    "aggregate_delta_advantage_pp", "best_fixed_member_sr",
    "aggregate_window_vs_best_member_pp"]
display(window_comparison[main_columns].rename(columns={
    "score_name": "uncertainty_horizon",
    "lower_u_baseline_sr": "aggregate_no_refinement_sr"}))'''),
    md("""## 6. Test alternative two-model uncertainty signals

The executed model is still whichever member has lower uncertainty. Only the score deciding
whether to refine changes: `minimum_u`, `(U0+U1)/2`, `maximum_u`, or `abs(U0-U1)`. Each signal gets
its own window sweep. Positive `delta_advantage_vs_source_pp` means that signal's best aggregation
window improves SR more than the source checkpoint's own best window."""),
    code(r'''from pnp.diversity import aggregation_gate_signal_window_sweep

gate_sweep, gate_best = aggregation_gate_signal_window_sweep(
    tables["selective_refinement_policy_pairs"])
source_reference = window_comparison[[
    "score_name", "source_window_delta_pp", "source_window_sr"]]
gate_comparison = gate_best[gate_best.score_name.isin(horizons)][[
    "score_name", "gate_signal", "lower", "upper", "n_refined", "selective_sr",
    "delta_pp", "delta_vs_best_fixed_pp"]].merge(
        source_reference, on="score_name", validate="many_to_one")
gate_comparison["delta_advantage_vs_source_pp"] = (
    gate_comparison.delta_pp - gate_comparison.source_window_delta_pp)
gate_comparison["window_sr_vs_source_window_pp"] = 100 * (
    gate_comparison.selective_sr - gate_comparison.source_window_sr)
signal_order = ["minimum_u", "mean_u", "maximum_u", "absolute_u_gap"]
gate_comparison["score_name"] = pd.Categorical(
    gate_comparison.score_name, categories=horizons, ordered=True)
gate_comparison["gate_signal"] = pd.Categorical(
    gate_comparison.gate_signal, categories=signal_order, ordered=True)
gate_comparison = gate_comparison.sort_values(["score_name", "gate_signal"])
signal_winners = gate_comparison.sort_values(
    ["score_name", "delta_pp"], ascending=[True, False]).groupby(
        "score_name", observed=True).head(1)
tables["alternative_aggregate_gate_signal_sweep"] = gate_sweep
tables["alternative_aggregate_gate_signal_best_windows"] = gate_comparison
tables["alternative_aggregate_gate_signal_winners"] = signal_winners

display(gate_comparison.rename(columns={
    "score_name": "uncertainty_horizon",
    "selective_sr": "aggregate_window_sr",
    "delta_pp": "aggregate_window_delta_pp",
    "delta_vs_best_fixed_pp": "aggregate_window_vs_best_member_pp"}))
print("Best two-model gating signal at each horizon")
display(signal_winners[[
    "score_name", "gate_signal", "lower", "upper", "n_refined", "selective_sr",
    "delta_pp", "source_window_delta_pp", "delta_advantage_vs_source_pp"]])'''),
    md("""## 7. Oracle ceilings and source-plus-member ensembles

The oracle rows are impossible selectors that know each rollout's outcome in advance. They answer
whether the recorded arms contain enough complementary successes for a real selector to exploit.
The four-arm member oracle succeeds if any of model 0/1 baseline/refinement succeeds; the six-arm
oracle also includes source baseline/refinement.

The source-plus-member experiments are implementable selection rules on the recorded trajectories:
choose the lowest-uncertainty policy from the named pair or trio, then refine that chosen policy
only when its uncertainty falls inside the swept window. Source + model 1 uses all 1,300 model-1
identities; source + model 0 and the three-policy option use the shared 650 identities. Positive
`pair_minus_optimal_source_pp` means the resulting pair policy beats the source checkpoint after
the source receives its own optimal uncertainty window at that same horizon. Every displayed SR
uses the full matched episode cohort, including episodes outside the refinement window."""),
    code(r'''from pnp.diversity import analyze_source_member_ensembles

# Exact episode-level oracle ceilings. These are diagnostics, not deployable policies.
oracle_episodes = paired[DIVERSITY_PAIR_KEYS + [
    "success_observed_m0", "success_refined_m0",
    "success_observed_m1", "success_refined_m1"]].merge(
        comparison[DIVERSITY_PAIR_KEYS + [
            "source_baseline_success", "source_refinement_success"]],
        on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
outcome_columns = [
    "source_baseline_success", "source_refinement_success",
    "success_observed_m0", "success_refined_m0",
    "success_observed_m1", "success_refined_m1"]
for column in outcome_columns:
    oracle_episodes[column] = oracle_episodes[column].astype(bool)

oracle_policies = {
    "source baseline": ["source_baseline_success"],
    "source refinement": ["source_refinement_success"],
    "model 0 baseline": ["success_observed_m0"],
    "model 1 baseline": ["success_observed_m1"],
    "2-member baseline oracle": ["success_observed_m0", "success_observed_m1"],
    "4-arm member oracle": ["success_observed_m0", "success_refined_m0",
                            "success_observed_m1", "success_refined_m1"],
    "source + members baseline oracle": ["source_baseline_success",
                                          "success_observed_m0", "success_observed_m1"],
    "all 6-arm oracle": outcome_columns,
}
source_baseline_sr = oracle_episodes.source_baseline_success.mean()
best_source_window_sr = window_comparison.source_window_sr.max()
oracle_summary = pd.DataFrame([{
    "policy": name, "n": len(oracle_episodes),
    "n_success": int(oracle_episodes[columns].any(axis=1).sum()),
    "success_rate": oracle_episodes[columns].any(axis=1).mean(),
    "delta_vs_source_baseline_pp": 100 * (
        oracle_episodes[columns].any(axis=1).mean() - source_baseline_sr),
    "delta_vs_best_source_window_pp": 100 * (
        oracle_episodes[columns].any(axis=1).mean() - best_source_window_sr),
} for name, columns in oracle_policies.items()])
tables["oracle_opportunity_episodes"] = oracle_episodes
tables["oracle_opportunity_summary"] = oracle_summary
print("Oracle opportunity (upper bounds; outcome knowledge is not deployable)")
oracle_display = oracle_summary[[
    "policy", "n_success", "n", "success_rate",
    "delta_vs_best_source_window_pp"]].copy()
oracle_display["success_rate"] *= 100
display(oracle_display.rename(columns={
    "success_rate": "oracle_sr_all_episodes_pct",
    "delta_vs_best_source_window_pp": "oracle_minus_optimal_source_pp"}))

# Source+m1 uses all 1,300 identities. Source+m0 and source+m0+m1 use the shared 650.
source_m1_full = analyze_source_member_ensembles(
    m1_full_analysis["member_refinement_pairs"],
    source_m1_analysis["member_refinement_pairs"],
    fixed_threshold=FIXED_THRESHOLD)
shared_source_members = analyze_source_member_ensembles(
    tables["member_refinement_pairs"], source_analysis["member_refinement_pairs"],
    fixed_threshold=FIXED_THRESHOLD)

source_member = {}
for name in source_m1_full:
    shared = shared_source_members[name]
    # The shared source+m1 duplicate is intentionally replaced by the full 1,300-row version.
    shared = shared[shared.ensemble.isin(["source_plus_m0", "source_plus_m0_m1"])]
    source_member[name] = pd.concat([source_m1_full[name], shared], ignore_index=True)
    tables[name] = source_member[name]

source_member_best = source_member["source_member_best_windows"]
source_member_overall = source_member["source_member_overall"]
source_m1_top = source_m1_analysis["member_refinement_top_windows"]
source_m1_reference = source_m1_top[
    source_m1_top["rank"].eq(1) & source_m1_top.score_name.isin(horizons)][[
        "score_name", "delta_pp", "selective_sr"]].rename(columns={
            "delta_pp": "source_window_delta_pp",
            "selective_sr": "source_window_sr"})
source_m1_reference["ensemble"] = "source_plus_m1"
shared_source_reference = pd.concat([
    window_comparison[["score_name", "source_window_delta_pp", "source_window_sr"]]
    .assign(ensemble=ensemble)
    for ensemble in ("source_plus_m0", "source_plus_m0_m1")], ignore_index=True)
source_member_reference = pd.concat(
    [source_m1_reference, shared_source_reference], ignore_index=True)
source_member_comparison = (
    source_member_best[source_member_best.score_name.isin(horizons)][[
        "ensemble", "member_index", "score_name", "lower", "upper", "n_refined",
        "selective_sr", "delta_pp", "delta_vs_best_fixed_pp"]]
    .merge(source_member_overall[source_member_overall.score_name.isin(horizons)][[
        "ensemble", "score_name", "n_pairs", "lower_u_baseline_sr",
        "best_fixed_member_sr"]],
        on=["ensemble", "score_name"], validate="one_to_one")
    .merge(source_member_reference, on=["ensemble", "score_name"], validate="one_to_one"))
source_member_comparison["delta_advantage_vs_source_pp"] = (
    source_member_comparison.delta_pp - source_member_comparison.source_window_delta_pp)
source_member_comparison["window_sr_vs_source_window_pp"] = 100 * (
    source_member_comparison.selective_sr - source_member_comparison.source_window_sr)
source_member_comparison["score_name"] = pd.Categorical(
    source_member_comparison.score_name, categories=horizons, ordered=True)
ensemble_order = ["source_plus_m1", "source_plus_m0", "source_plus_m0_m1"]
source_member_comparison["ensemble"] = pd.Categorical(
    source_member_comparison.ensemble, categories=ensemble_order, ordered=True)
source_member_comparison = source_member_comparison.sort_values(
    ["ensemble", "score_name"])
tables["source_member_best_window_comparison"] = source_member_comparison
source_member_display = source_member_comparison[[
    "ensemble", "score_name", "n_pairs", "lower", "upper", "n_refined",
    "selective_sr", "source_window_sr", "window_sr_vs_source_window_pp"]].copy()
source_member_display["selective_sr"] *= 100
source_member_display["source_window_sr"] *= 100
display(source_member_display.rename(columns={
    "ensemble": "pair",
    "score_name": "uncertainty_horizon",
    "n_pairs": "episodes_in_sr_denominator",
    "lower": "window_lower",
    "upper": "window_upper",
    "n_refined": "episodes_refined",
    "selective_sr": "pair_sr_all_episodes_pct",
    "source_window_sr": "optimal_source_sr_all_episodes_pct",
    "window_sr_vs_source_window_pp": "pair_minus_optimal_source_pp"}))'''),
    md("""## 8. Supporting aggregation details

The table below retains the five best windows per horizon so nearby alternatives can be inspected.
The fixed `0.03` gate is shown only as a secondary predeclared comparison; it is not the main
window-sweep result."""),
    code(r'''top = tables["selective_refinement_top_windows"]
display(top[top.score_name.isin(horizons) & (top["rank"] <= 5)][[
    "score_name", "rank", "lower", "upper", "n_refined", "selective_sr",
    "delta_pp", "delta_vs_best_fixed_pp", "selected_F_to_S", "selected_S_to_F"]])

overall = tables["selective_refinement_overall"]
print("Secondary comparison: fixed U >= 0.03 gate")
display(overall[overall.score_name.isin(horizons)][[
    "score_name", "best_fixed_member_sr", "lower_u_baseline_sr",
    "fixed_threshold_sr", "fixed_delta_vs_lower_u_pp",
    "fixed_delta_vs_best_fixed_pp", "fixed_vs_best_ci_low_pp",
    "fixed_vs_best_ci_high_pp", "n_refined_fixed", "fixed_F_to_S", "fixed_S_to_F"]])'''),
    md("## 9. Save tables and figures"),
    code(r'''for name, frame in tables.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)
paths = diversity_selective_refinement_figures(tables, OUTPUT / "figures")
primary_names = {"source_vs_aggregate_best_windows.png",
                 "alternative_aggregate_gate_signals.png",
                 "oracle_opportunity.png",
                 "source_member_best_windows.png"} | {
    f"source_vs_aggregate_window_sweep_{name}.png" for name in horizons}
print("Primary source-versus-aggregation figures")
for path in paths:
    if path.name in primary_names:
        print(path.name)
        display(Image(filename=str(path)))
print(f"Saved {len(paths) - len(primary_names)} additional diagnostic figures without displaying them.")'''),
    md("## 10. Concise readout"),
    code(r'''for _, row in window_comparison.iterrows():
    print(str(row.score_name).replace("_", " "))
    print("  source window:    %5.1f%% SR (%+.2f pp over source baseline)" %
          (100 * row.source_window_sr, row.source_window_delta_pp))
    print("  aggregate window: %5.1f%% SR (%+.2f pp over lower-U baseline)" %
          (100 * row.aggregate_window_sr, row.aggregate_window_delta_pp))
    print("  aggregate delta advantage over source: %+.2f pp" %
          row.aggregate_delta_advantage_pp)
four_arm = oracle_summary[oracle_summary.policy.eq("4-arm member oracle")].iloc[0]
six_arm = oracle_summary[oracle_summary.policy.eq("all 6-arm oracle")].iloc[0]
print("\nOracle ceilings (not deployable)")
print("  four member arms: %.1f%% SR (%+.2f pp versus best source window)" %
      (100 * four_arm.success_rate, four_arm.delta_vs_best_source_window_pp))
print("  source + all member arms: %.1f%% SR (%+.2f pp versus best source window)" %
      (100 * six_arm.success_rate, six_arm.delta_vs_best_source_window_pp))
best_pair = source_member_comparison.sort_values(
    "window_sr_vs_source_window_pp", ascending=False).iloc[0]
print("Best source+member result")
print("  %s, %s: %.1f%% SR (%+.2f pp versus source's window at that horizon)" %
      (best_pair.ensemble, str(best_pair.score_name).replace("_", " "),
       100 * best_pair.selective_sr, best_pair.window_sr_vs_source_window_pp))
print("\nFirst chunk is deployable at the initial observation; later horizons are post-hoc here.")'''),
], "21_analyze_diversity_selective_refinement_v2.ipynb")


def chunk_selector_worker_notebook(shard_index: int):
    return notebook([
        md(f"""# 22 - Online per-chunk selector worker {shard_index}

This is the causal online aggregation experiment. At every 10-action decision chunk, both arms
generate two clean candidates from the same live observation and execute the lower-uncertainty
candidate:

- `source, 2 queries`: two independent source-checkpoint queries;
- `source + model 1`: one source query and one v2 model-1 query.

Both use K=5 uncertainty at Euler steps `(3,4)` and matched candidate-slot seeds. The two arms
start from exactly the same LIBERO-PRO identity, although their states may diverge after executing
different chunks. Candidate uncertainties, choices, clean-action disagreement metrics, and raw
candidate chunks are logged. No Supabase migration is required.

This notebook is shard {shard_index} of 4. It is resumable and uses 10 episodes per task, so all
four workers together produce 1,300 matched identities per arm. Before committing compute, worker
0 can be run once with `EPISODE_LIMIT=1`; after those two rollouts succeed, set it back to `None`
and rerun. Those smoke-test rows are reused rather than repeated. Use an A100 if available because
source+m1 keeps two pi0.5 policies resident simultaneously; an L4 is not assumed to fit."""),
        md("## 1. Setup a fresh GPU runtime"), code(bootstrap(extras="sim", setup_env=True)),
        md("## 2. Configuration and resumable collection"),
        code(f'''from pathlib import Path
from google.colab import drive
from pnp.config import PI05_REPO_ID
from pnp.diversity import (DIVERSITY_CHUNK_SELECTOR_EXPERIMENT,
    DIVERSITY_V2_EXPERIMENT_PREFIX, load_bootstrap_manifest,
    run_diversity_chunk_selector_worker)

drive.mount("/content/drive")

EPISODES_PER_TASK = 10
SHARD_COUNT = 4
SHARD_INDEX = {shard_index}
EPISODE_LIMIT = None  # optional worker-0 smoke test: set 1, run, then restore None
EXPERIMENT = DIVERSITY_CHUNK_SELECTOR_EXPERIMENT
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/pnp_diversity_v2/bootstrap_manifest_finetuned_v2.json")
manifest = load_bootstrap_manifest(MANIFEST_PATH)
assert manifest["source_model"] == PI05_REPO_ID, manifest["source_model"]
SOURCE_MODEL_REVISION = manifest["source_model_revision"]
assert SOURCE_MODEL_REVISION, "v2 manifest is missing source_model_revision"

print({{"experiment": EXPERIMENT, "episodes_per_task": EPISODES_PER_TASK,
       "shard_count": SHARD_COUNT, "shard_index": SHARD_INDEX,
       "episode_limit": EPISODE_LIMIT,
       "manifest_hash": manifest["manifest_hash"],
       "source_model_revision": SOURCE_MODEL_REVISION}})
run_diversity_chunk_selector_worker(
    episodes_per_task=EPISODES_PER_TASK,
    episode_limit=EPISODE_LIMIT,
    shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
    manifest_hash=manifest["manifest_hash"],
    source_model_revision=SOURCE_MODEL_REVISION,
    diversity_experiment_prefix=DIVERSITY_V2_EXPERIMENT_PREFIX,
    experiment=EXPERIMENT)'''),
    ], f"22_diversity_chunk_selector_worker_{shard_index}.ipynb")


ANALYZE_CHUNK_SELECTOR = notebook([
    md("""# 23 - Analyze online per-chunk source+m1 aggregation

Primary question: does choosing between source and model 1 at every live decision chunk outperform
the matched control that chooses between two stochastic source queries? All SR comparisons use
exact episode identities. The old single-source K=5 `(3,4)` rollout is also matched as context.

Action-diversity metrics compare the two original clean candidate predictions before selection.
This tests whether model 1 contributes meaningfully more behavioral diversity than simply
re-querying the source model."""),
    md("## 1. Setup"), code(bootstrap(extras="analysis", setup_env=False)),
    md("## 2. Fetch the exact matched online cohort"),
    code(r'''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from pnp.config import Method
from pnp.diversity import (DIVERSITY_CHUNK_SELECTOR_EXPERIMENT,
    DIVERSITY_PAIR_KEYS, analyze_diversity_chunk_selector,
    diversity_chunk_selector_figures, fetch_diversity_chunk_selector)
from pnp.experiments import PRO_EXPANDED_EXPERIMENT
from pnp.store import SupabaseStore

EXPERIMENT = DIVERSITY_CHUNK_SELECTOR_EXPERIMENT
REQUIRE_FULL_COHORT = False  # set True only after workers 0,1,2,3 finish
OUTPUT = Path("diversity_chunk_selector_outputs")
OUTPUT.mkdir(exist_ok=True)

store = SupabaseStore()
all_rollouts = fetch_diversity_chunk_selector(store, experiment=EXPERIMENT)
counts = (all_rollouts.drop_duplicates(DIVERSITY_PAIR_KEYS + ["method"])
          .groupby(DIVERSITY_PAIR_KEYS).method.nunique().rename("n_arms").reset_index())
complete_keys = counts[counts.n_arms.eq(2)][DIVERSITY_PAIR_KEYS]
rollouts = all_rollouts.merge(complete_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
arm_counts = rollouts.groupby("method").size().rename("n").reset_index()
display(arm_counts)
assert arm_counts.n.nunique() == 1
N_PAIRS = int(arm_counts.n.iloc[0])
if REQUIRE_FULL_COHORT:
    assert N_PAIRS == 1300, f"Full cohort requires 1,300 pairs; found {N_PAIRS}"
print({"analysis_mode": "FULL" if REQUIRE_FULL_COHORT else "PARTIAL PREVIEW",
       "matched_episode_identities": N_PAIRS,
       "incomplete_identities_ignored": int(counts.n_arms.lt(2).sum())})

tables = analyze_diversity_chunk_selector(rollouts)'''),
    md("## 3. Match the old single-source baseline"),
    code(r'''source_rows = pd.DataFrame(store.fetch_all(
    "rollouts", "rollout_id,suite,task_idx,episode_idx,init_state_hash,method,status,success,"
    "max_steps,pnp_k,pnp_step_indices",
    configure=lambda query: query.eq("experiment", PRO_EXPANDED_EXPERIMENT),
    order_by=("rollout_id",)))
source_rows = source_rows[
    source_rows.status.eq("completed") & source_rows.method.eq(Method.UNCERTAINTY) &
    source_rows.pnp_k.eq(5) &
    source_rows.pnp_step_indices.apply(lambda value: tuple(value or []) == (3, 4))]
assert not source_rows.duplicated(DIVERSITY_PAIR_KEYS).any()
paired = tables["chunk_selector_paired_episodes"]
source = source_rows[DIVERSITY_PAIR_KEYS + ["success", "max_steps"]].rename(
    columns={"success": "source_baseline_success", "max_steps": "source_max_steps"})
paired_with_source = paired.merge(
    source, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
assert len(paired_with_source) == len(paired), (
    f"Source baseline matched {len(paired_with_source)}/{len(paired)} identities")
assert (paired_with_source.source_max_steps == paired_with_source[
    f"max_steps_{Method.CHUNK_SOURCE_SOURCE}"]).all(), (
    "The historical source baseline uses a different episode horizon")
source_success = paired_with_source.source_baseline_success.astype(bool)
control_success = paired_with_source[
    f"success_{Method.CHUNK_SOURCE_SOURCE}"].astype(bool)
treatment_success = paired_with_source[
    f"success_{Method.CHUNK_SOURCE_M1}"].astype(bool)
source_comparison = pd.DataFrame([{
    "n_matched_episodes": len(paired_with_source),
    "single_source_sr": source_success.mean(),
    "source_two_queries_sr": control_success.mean(),
    "source_plus_m1_sr": treatment_success.mean(),
    "two_queries_minus_single_source_pp": 100 * (
        control_success.mean() - source_success.mean()),
    "source_m1_minus_single_source_pp": 100 * (
        treatment_success.mean() - source_success.mean()),
    "source_m1_minus_two_queries_pp": 100 * (
        treatment_success.mean() - control_success.mean()),
}])
tables["chunk_selector_source_baseline"] = pd.DataFrame([{
    "n": len(paired_with_source), "source_baseline_sr": source_success.mean()}])
tables["chunk_selector_paired_with_source"] = paired_with_source
tables["chunk_selector_source_comparison"] = source_comparison
display(source_comparison)'''),
    md("## 4. Paired outcome result"),
    code(r'''print("Source+m1 versus the two-source-query control")
display(tables["chunk_selector_paired_summary"])
display(tables["chunk_selector_by_suite"][[
    "suite", "n_pairs", "source_requery_sr", "source_m1_sr",
    "source_m1_minus_requery_pp", "F_to_S", "S_to_F"]])'''),
    md("""## 5. Are source+m1 predictions more diverse?

These metrics compare the two clean policy-space candidate chunks before the lower-U choice.
Higher L2/lower cosine means greater prediction diversity. `candidate1_selected_rate` reports how
often the second source query or model 1 had lower uncertainty."""),
    code(r'''diversity = tables["chunk_selector_diversity_overall"].copy()
diversity["pair"] = diversity.method.map({
    Method.CHUNK_SOURCE_SOURCE: "source vs source",
    Method.CHUNK_SOURCE_M1: "source vs model 1"})
display(diversity[[
    "pair", "n_chunks", "candidate1_selected_rate", "candidate_u_gap_mean",
    "action_l2_mean_mean", "action_l2_normalized_mean", "action_cosine_mean",
    "gripper_sign_disagreement_mean"]])

wide = diversity.set_index("pair")
diversity_comparison = pd.DataFrame([{
    "metric": "clean_action_l2",
    "source_vs_source": wide.loc["source vs source", "action_l2_mean_mean"],
    "source_vs_model1": wide.loc["source vs model 1", "action_l2_mean_mean"],
    "source_m1_over_requery_ratio": (
        wide.loc["source vs model 1", "action_l2_mean_mean"] /
        wide.loc["source vs source", "action_l2_mean_mean"]),
}, {
    "metric": "normalized_clean_action_l2",
    "source_vs_source": wide.loc["source vs source", "action_l2_normalized_mean"],
    "source_vs_model1": wide.loc["source vs model 1", "action_l2_normalized_mean"],
    "source_m1_over_requery_ratio": (
        wide.loc["source vs model 1", "action_l2_normalized_mean"] /
        wide.loc["source vs source", "action_l2_normalized_mean"]),
}])
tables["chunk_selector_diversity_comparison"] = diversity_comparison
display(diversity_comparison)
print("Raw candidate chunks are preserved in each rollout's generated_chunks_path blob.")'''),
    md("## 6. Save tables and plots"),
    code(r'''for name, frame in tables.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)
paths = diversity_chunk_selector_figures(tables, OUTPUT / "figures")
for path in paths:
    print(path.name)
    display(Image(filename=str(path)))'''),
    md("## 7. Concise readout"),
    code(r'''result = source_comparison.iloc[0]
diversity_result = diversity_comparison.set_index("metric")
print(f"Matched episodes: {int(result.n_matched_episodes)}")
print(f"Single source:       {result.single_source_sr:.1%}")
print(f"Source, two queries: {result.source_two_queries_sr:.1%} "
      f"({result.two_queries_minus_single_source_pp:+.2f} pp vs single source)")
print(f"Source + model 1:    {result.source_plus_m1_sr:.1%} "
      f"({result.source_m1_minus_two_queries_pp:+.2f} pp vs two-query control)")
print("Source+m1 / source-requery normalized action-diversity ratio: %.2fx" %
      diversity_result.loc["normalized_clean_action_l2",
                           "source_m1_over_requery_ratio"])'''),
], "23_analyze_diversity_chunk_selector.ipynb")


def main():
    write_notebook(ROOT / "notebooks" / "17_train_diverse_pi05.ipynb", TRAIN)
    write_notebook(ROOT / "notebooks" / "17_train_diverse_pi05_v2.ipynb", TRAIN_V2)
    for member_index in (0, 1):
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"18_diversity_signal_model_{member_index}.ipynb",
                       collection_notebook(member_index))
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"18_diversity_signal_v2_model_{member_index}.ipynb",
                       collection_v2_notebook(member_index))
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"20_diversity_selective_refinement_v2_model_{member_index}.ipynb",
                       selective_refinement_v2_notebook(member_index))
    write_notebook(ROOT / "notebooks" / "19_analyze_diversity_signal.ipynb", ANALYZE)
    write_notebook(ROOT / "notebooks" / "19_analyze_diversity_signal_v2.ipynb", ANALYZE_V2)
    write_notebook(ROOT / "notebooks" /
                   "21_analyze_diversity_selective_refinement_v2.ipynb",
                   ANALYZE_SELECTIVE_REFINEMENT_V2)
    for shard_index in range(4):
        write_notebook(ROOT / "notebooks" / "workers" /
                       f"22_diversity_chunk_selector_worker_{shard_index}.ipynb",
                       chunk_selector_worker_notebook(shard_index))
    write_notebook(ROOT / "notebooks" / "23_analyze_diversity_chunk_selector.ipynb",
                   ANALYZE_CHUNK_SELECTOR)


if __name__ == "__main__":
    main()
