"""Generate stable, thin Colab launchers; experiment logic stays in pnp.experiments."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "workers"
SHARD_COUNT = 6
ROLLOUT_BATCH_SIZE = 2

BOOTSTRAP = '''import os
import subprocess
import sys
from google.colab import userdata

for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "HF_TOKEN"):
    os.environ[key] = userdata.get(key)

gh_pat = userdata.get("GH_PAT")
repo_dir = "/content/cs159-sp26"
repo_url = f"https://{gh_pat}@github.com/ArjunS07/cs159-sp26.git"

if not os.path.isdir(os.path.join(repo_dir, ".git")):
    subprocess.run(["git", "clone", "--branch", "main", repo_url, repo_dir], check=True)
else:
    subprocess.run(["git", "-C", repo_dir, "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", "main"], check=True)
    subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only", "origin", "main"], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo_dir}/pnp-vla[sim]",
], check=True)

# Editable installs add a .pth file that a fresh interpreter would process at startup. This
# notebook interpreter was already running during pip, so expose the source tree immediately.
package_dir = f"{repo_dir}/pnp-vla"
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

import pnp
print("Loaded pnp from:", pnp.__file__)
'''

ENV_SETUP = '''from pnp.env_setup import setup_environment
setup_environment()  # If the runtime restarts, use Run all again.
'''


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
            cell("code", ENV_SETUP),
            cell("code", run),
        ],
    }


def experiment_notebook() -> dict:
    """Human-facing launcher; matrices and execution live exclusively in pnp.experiments."""
    run = f'''from pnp.experiments import run_libero_hybrid_worker, run_libero_pro_worker

SHARD_COUNT = 1
SHARD_INDEX = 0
ROLLOUT_BATCH_SIZE = {ROLLOUT_BATCH_SIZE}
RUN_LIBERO = True
RUN_LIBERO_PRO = False

if RUN_LIBERO:
    run_libero_hybrid_worker(shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
                             rollout_batch_size=ROLLOUT_BATCH_SIZE)
if RUN_LIBERO_PRO:
    run_libero_pro_worker(shard_count=SHARD_COUNT, shard_index=SHARD_INDEX,
                          rollout_batch_size=ROLLOUT_BATCH_SIZE)
'''
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"name": "01_run_experiments.ipynb",
                     "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("markdown", "# Run LIBERO experiments\n\nThin launcher for the package-owned experiment plans.\n"),
            cell("code", BOOTSTRAP), cell("code", ENV_SETUP), cell("code", run),
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for benchmark in ("libero", "libero_pro"):
        for shard_index in range(SHARD_COUNT):
            stem = "libero_worker" if benchmark == "libero" else "libero_pro_worker"
            path = OUT / f"{stem}_{shard_index}.ipynb"
            path.write_text(json.dumps(
                notebook(shard_index, benchmark=benchmark), indent=1) + "\n")
            print(path.relative_to(ROOT))
    main_path = ROOT / "notebooks" / "01_run_experiments.ipynb"
    main_path.write_text(json.dumps(experiment_notebook(), indent=1) + "\n")
    print(main_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
