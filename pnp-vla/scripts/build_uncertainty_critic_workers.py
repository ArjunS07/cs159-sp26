"""Build the four Colab workers for same-observation uncertainty-critic data."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "notebooks" / "workers"


def notebook(shard_index: int) -> dict:
    title = (
        f"# 45 - Same-observation uncertainty-critic worker {shard_index}\n\n"
        f"Shard {shard_index} of 4 over 800 standard-LIBERO development trajectories "
        "(init-state indices 20-39 for every task). At prediction chunks 0, 2, 4, "
        "and 8, this records six independently seeded ordinary action predictions "
        "from the same observation, their U10/U20/U50 labels, and intermediate P&P "
        "features. Only candidate 0 is executed, using exactly 10 actions before "
        "replanning; there is no refinement and no LIBERO-PRO data. Init-state "
        "indices 20-35 are critic-training data and 36-39 are held-out validation. "
        "Set `EPISODE_LIMIT = 1` for a resume-safe smoke test, then restore `None`."
    )
    bootstrap = (
        "EXTRAS = 'sim'\n"
        "SETUP_ENV = True\n"
        "import urllib.request\n"
        "exec(urllib.request.urlopen("
        "'https://raw.githubusercontent.com/ArjunS07/cs159-sp26/main/"
        "pnp-vla/scripts/colab_bootstrap.py').read().decode())"
    )
    run = f'''from pnp.uncertainty_critic import (
    EXPERIMENT, CANDIDATE_COUNT, TARGET_CHUNKS, run_worker)

SHARD_COUNT = 4
SHARD_INDEX = {shard_index}
EPISODE_LIMIT = None  # set to 1 for a smoke test; that row is reused on the full run

print({{
    'experiment': EXPERIMENT,
    'shard_count': SHARD_COUNT,
    'shard_index': SHARD_INDEX,
    'episode_limit': EPISODE_LIMIT,
    'full_trajectories_in_shard': 200,
    'candidate_predictions_per_state': CANDIDATE_COUNT,
    'target_chunk_indices': list(TARGET_CHUNKS),
    'maximum_candidate_examples_in_shard': 4800,
    'executed_candidate': 0,
    'n_action_steps': 10,
    'primary_critic_input': 'ordinary pre-refinement action chunk',
    'labels': ['U10', 'U20', 'U50'],
    'split': 'init states 20-35 train; 36-39 validation; 0-19 untouched eval',
}})
run_worker(
    shard_count=SHARD_COUNT,
    shard_index=SHARD_INDEX,
    experiment=EXPERIMENT,
    episode_limit=EPISODE_LIMIT,
)'''
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [title]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [bootstrap]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [run]},
        ],
        "metadata": {
            "colab": {
                "name": f"45_uncertainty_critic_candidates_worker_{shard_index}.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    WORKERS.mkdir(parents=True, exist_ok=True)
    for shard_index in range(4):
        path = WORKERS / f"45_uncertainty_critic_candidates_worker_{shard_index}.ipynb"
        path.write_text(json.dumps(notebook(shard_index), indent=1) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
