"""Generate stable, thin Colab launchers; experiment logic stays in pnp.experiments."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "workers"
SHARD_COUNT = 6

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


def notebook(shard_index: int) -> dict:
    run = f'''from pnp.experiments import run_libero_hybrid_worker

SHARD_COUNT = {SHARD_COUNT}
SHARD_INDEX = {shard_index}
run_libero_hybrid_worker(shard_count=SHARD_COUNT, shard_index=SHARD_INDEX)
'''
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"libero_worker_{shard_index}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("markdown", f"# LIBERO collection worker {shard_index}/{SHARD_COUNT}\n\n"
                 "Stable launcher: all mutable experiment logic is pulled from `pnp.experiments`.\n"),
            cell("code", BOOTSTRAP),
            cell("code", ENV_SETUP),
            cell("code", run),
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for shard_index in range(SHARD_COUNT):
        path = OUT / f"libero_worker_{shard_index}.ipynb"
        path.write_text(json.dumps(notebook(shard_index), indent=1) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
