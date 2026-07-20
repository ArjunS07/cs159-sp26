"""Clean-action verifier models, datasets, training, and paired collection helpers."""

from .data import (
    CleanChunkExample,
    build_clean_chunk_examples,
    hard_task_keys,
    heldout_task_split,
    known_task_split,
    load_clean_chunk_examples,
)
from .model import CleanChunkVerifier, VerifierOutput
from .collection import (
    SimulatorSnapshot,
    candidate_group_id,
    capture_snapshot,
    restore_snapshot,
    validate_snapshot_replay,
)
from .train import (
    TemperatureScaler,
    VerifierTrainConfig,
    calibrate_temperature,
    evaluate_verifier,
    train_verifier,
)

__all__ = [
    "CleanChunkExample",
    "CleanChunkVerifier",
    "TemperatureScaler",
    "VerifierOutput",
    "VerifierTrainConfig",
    "build_clean_chunk_examples",
    "calibrate_temperature",
    "evaluate_verifier",
    "hard_task_keys",
    "heldout_task_split",
    "known_task_split",
    "load_clean_chunk_examples",
    "train_verifier",
    "SimulatorSnapshot",
    "candidate_group_id",
    "capture_snapshot",
    "restore_snapshot",
    "validate_snapshot_replay",
]
