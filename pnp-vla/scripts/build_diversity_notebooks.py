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

It can be run as a preview after shards 0-1: analysis is restricted to identities completed in
all four member/method arms. Set `REQUIRE_FULL_COHORT=True` only after shards 0-3 finish.

Prefix, individual-chunk, and full-episode summaries are also analyzed. Only the first-chunk rule
is directly deployable from these independently simulated trajectories. Later scores are useful
post-hoc evidence about predictiveness and headroom, but they are not presented as an online
policy. The best in-sample window is exploratory; leave-one-suite-out (LOSO) window selection is
reported separately to reduce threshold-selection optimism."""),
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
OUTPUT = Path("diversity_selective_refinement_v2_outputs")
OUTPUT.mkdir(exist_ok=True)

store = SupabaseStore()
all_rollouts, all_steps = fetch_diversity_selective_refinement(
    store, experiment_prefix=EXPERIMENT_PREFIX)
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

# Run the same horizon/AUC/window analysis on the source checkpoint, using only exact
# current/source identities and only source episodes with both baseline and refinement arms.
matched_keys = comparison[DIVERSITY_PAIR_KEYS]
source_analysis_rollouts = pd.concat(
    [source_observed_rows, source_refined_rows], ignore_index=True).merge(
        matched_keys, on=DIVERSITY_PAIR_KEYS, validate="many_to_one")
source_observed_ids = source_analysis_rollouts[
    source_analysis_rollouts.method.eq(Method.UNCERTAINTY)].rollout_id.astype(str).tolist()
source_step_rows = []
for start in range(0, len(source_observed_ids), 100):
    batch = source_observed_ids[start:start + 100]
    source_step_rows.extend(store.fetch_all(
        "pnp_euler_steps", "rollout_id,chunk_idx,euler_step,u_mean",
        configure=lambda query, ids=batch: query.in_("rollout_id", ids),
        order_by=("rollout_id",)))
source_steps = pd.DataFrame(source_step_rows)
source_analysis = analyze_checkpoint_refinement(
    source_analysis_rollouts, source_steps, checkpoint_name="source_checkpoint")
for name, frame in source_analysis.items():
    tables[name.replace("member_refinement", "source_checkpoint_refinement")] = frame

source_horizons = source_analysis["member_refinement_overall"]
print("Shared source checkpoint: matched refinement delta and failure AUC")
display(source_horizons[source_horizons.score_name.isin(horizons)][[
    "score_name", "n_pairs", "baseline_sr", "refinement_sr", "delta_pp",
    "delta_ci_low_pp", "delta_ci_high_pp", "F_to_S", "S_to_F", "failure_auc"]])
print("Shared source checkpoint: top exploratory uncertainty windows")
source_top = source_analysis["member_refinement_top_windows"]
display(source_top[source_top.score_name.isin(horizons) & (source_top["rank"] <= 5)][[
    "score_name", "rank", "lower", "upper", "n_refined", "selective_sr",
    "delta_pp", "selected_F_to_S", "selected_S_to_F"]])'''),
    md("## 5. Two-model lower-uncertainty aggregation"),
    code(r'''overall = tables["selective_refinement_overall"]
columns = ["score_name", "interpretation", "n_pairs", "best_fixed_member_sr",
           "lower_u_baseline_sr", "lower_u_delta_vs_best_fixed_pp",
           "n_model_discordant", "lower_u_selector_accuracy_discordant",
           "lower_u_selector_win_auc", "lower_u_refine_all_sr",
           "n_refined_fixed", "fixed_threshold_sr",
           "fixed_delta_vs_lower_u_pp", "fixed_delta_ci_low_pp",
           "fixed_delta_ci_high_pp", "fixed_F_to_S", "fixed_S_to_F"]
print("Lower-U selection plus primary 0.03 refinement gate")
display(overall[overall.score_name.isin(horizons)][columns])
print("Individual chunks (chunk 0 is deployable; later chunks are post-hoc here)")
display(overall[overall.signal_kind == "individual_chunk"][columns])'''),
    md("""## 6. Aggregated exploratory windows and cross-validated estimate

The top-window table is selected and evaluated on the same data, so it is optimistic. LOSO
chooses a window on 12 suites and applies it once to the held-out suite."""),
    code(r'''top = tables["selective_refinement_top_windows"]
print("Top aggregated in-sample windows")
display(top[(top.score_name.isin(horizons)) & (top["rank"] <= 5)][
    ["score_name", "rank", "lower", "upper", "n_refined", "selective_sr",
     "delta_pp", "selected_F_to_S", "selected_S_to_F"]])
print("Leave-one-suite-out aggregated window selection")
display(tables["selective_refinement_loso_summary"])
print("Primary first-chunk fixed-threshold result by suite")
by_suite = tables["selective_refinement_fixed_by_suite"]
display(by_suite[by_suite.score_name == "first_chunk"][[
    "suite", "n_pairs", "lower_u_baseline_sr", "n_refined_fixed",
    "fixed_threshold_sr", "fixed_delta_vs_lower_u_pp", "fixed_F_to_S", "fixed_S_to_F"]])'''),
    md("## 7. Save tables and figures"),
    code(r'''for name, frame in tables.items():
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)
paths = diversity_selective_refinement_figures(tables, OUTPUT / "figures")
for path in paths:
    print(path.name)
    display(Image(filename=str(path)))'''),
    md("## 8. Concise readout"),
    code(r'''overall = tables["selective_refinement_overall"].set_index("score_name")
loso = tables["selective_refinement_loso_summary"].set_index("score_name")
for score_name in ("first_chunk", "full_episode"):
    row, cv = overall.loc[score_name], loso.loc[score_name]
    print(score_name.replace("_", " "))
    print("  lower-U baseline: %.1f%%" % (100 * row.lower_u_baseline_sr))
    print("  fixed U >= %.2f: %.1f%% (%+.1f pp; F->S=%d, S->F=%d)" %
          (row.fixed_threshold, 100 * row.fixed_threshold_sr,
           row.fixed_delta_vs_lower_u_pp, row.fixed_F_to_S, row.fixed_S_to_F))
    print("  refine selected model always: %.1f%% (%+.1f pp)" %
          (100 * row.lower_u_refine_all_sr, row.refine_all_delta_vs_lower_u_pp))
    print("  LOSO window: %.1f%% (%+.1f pp, CI [%+.1f, %+.1f])" %
          (100 * cv.loso_selective_sr, cv.delta_pp,
           cv.delta_ci_low_pp, cv.delta_ci_high_pp))
print("\nInterpretation: first chunk is deployable; full episode and later chunks are post-hoc.")'''),
], "21_analyze_diversity_selective_refinement_v2.ipynb")


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


if __name__ == "__main__":
    main()
