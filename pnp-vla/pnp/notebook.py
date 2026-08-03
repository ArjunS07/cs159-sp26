"""Post-install helpers for the thin Colab notebooks.

The clone/install cell cannot import from ``pnp`` (the package isn't installed
yet), so that stays a raw bootstrap. Everything *after* install -- opening a
store, resolving the device, creating the run's output directory, and guarding
the sealed confirmatory cohort -- is duplicated boilerplate that lives here so
notebooks reduce to a couple of calls.

Imports cleanly without the GPU/sim stack: ``torch`` is imported lazily inside
``resolve_device`` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import SupabaseStore

#: The one sealed cohort whose outcomes must never leak into training/dev work.
CONFIRMATORY_EXPERIMENT = "verifier-v2-pro-confirmatory"


def package_root() -> Path:
    """Repo-local ``pnp-vla`` directory (parent of the installed ``pnp`` package)."""
    return Path(__file__).resolve().parents[1]


def resolve_device(device: Any = None):
    """Return a ``torch.device``; default picks CUDA when available."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_dir(subdir: str, *, base: str | Path | None = None) -> Path:
    """Create and return ``<package_root>/analysis_outputs/<subdir>``."""
    root = Path(base) if base is not None else package_root()
    path = root / "analysis_outputs" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class NotebookContext:
    """The handful of objects every training/analysis notebook opens at setup."""
    store: SupabaseStore
    device: Any  # torch.device (kept untyped so this module imports without torch)
    output: Path


def setup(subdir: str, *, device: Any = None, store: SupabaseStore | None = None,
          base: str | Path | None = None) -> NotebookContext:
    """Open a store, resolve the device, and create the run's output directory."""
    return NotebookContext(
        store=store or SupabaseStore(),
        device=resolve_device(device),
        output=output_dir(subdir, base=base),
    )


def sealed_identities(store: SupabaseStore, experiment: str = CONFIRMATORY_EXPERIMENT,
                      *, min_groups: int = 120) -> tuple[set, list[dict]]:
    """Return the sealed cohort's episode identities and raw group rows.

    Only group identities are read -- ``verifier_candidates`` (the outcomes) are
    never queried -- so callers can exclude these identities from training/dev
    data without ever seeing the sealed labels.
    """
    rows = store.fetch_all(
        "verifier_candidate_groups",
        "candidate_group_id,benchmark,suite,task_idx,episode_idx",
        configure=lambda query: query.eq("experiment", experiment))
    assert len(rows) >= min_groups, len(rows)
    identities = {
        (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
        for row in rows
    }
    return identities, rows


def drop_sealed(examples, identities) -> list:
    """Exclude any example whose episode identity is in the sealed set."""
    from .verifier import exclude_episode_identities

    return exclude_episode_identities(examples, identities)
