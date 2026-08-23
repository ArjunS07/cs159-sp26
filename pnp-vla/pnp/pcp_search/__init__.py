"""PCP-search data collection and task-selection infrastructure.

This package is deliberately independent of the legacy Q-corrector modules.  It owns the
immutable rollout manifest, the Bellman/RL-token collection contract, and the Colab worker entry
point for the new critic.
"""

from .manifest import ManifestItem, RolloutManifest
from .task_selection import (
    INITIAL_ROLLOUT_BUDGET,
    build_initial_manifest,
    build_next_tranche_manifest,
    initial_task_allocation,
)
from .collection import run_pcp_search_worker

__all__ = [
    "INITIAL_ROLLOUT_BUDGET",
    "ManifestItem",
    "RolloutManifest",
    "build_initial_manifest",
    "build_next_tranche_manifest",
    "initial_task_allocation",
    "run_pcp_search_worker",
]
