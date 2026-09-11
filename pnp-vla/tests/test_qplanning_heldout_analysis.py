import numpy as np
import pandas as pd

from analysis.qplanning_heldout import (
    ARMS, extract_qplanning_boundaries, pair_against_stock,
    q_episode_features, q_prefix_features, quantile_gate_sweep,
    validate_qplanning_heldout_cohort,
)
from pnp.config import Method
from pnp.diversity import DIVERSITY_PAIR_KEYS


def _telemetry(offset=0.0):
    selections = []
    for chunk in range(2):
        q = np.linspace(0.0, 1.0, 64) + offset + chunk
        selections.append({
            "q_values": q.tolist(), "best_index": 63,
            "elite_indices": list(range(48, 64)),
            "elite_weights": np.full(16, 1 / 16).tolist(),
            "first10_diversity": {
                "actions_compared": 10, "mean_pairwise_rms": .1 + chunk,
                "max_pairwise_rms": .2 + chunk, "mean_pairwise_cosine": .9},
            "full50_diversity": {
                "actions_compared": 50, "mean_pairwise_rms": .3 + chunk,
                "max_pairwise_rms": .4 + chunk, "mean_pairwise_cosine": .8},
            "inference_ms": 12.0, "n_vf_evals": 192,
        })
    fields = selections[0].keys()
    return {"qplanning": {field: [item[field] for item in selections]
                           for field in fields}}


def _rows():
    rows = []
    for episode in range(3):
        for method in ARMS:
            rows.append({
                "suite": "libero_object_temp_x0.1", "task_idx": episode,
                "episode_idx": 0, "init_state_hash": f"h{episode}",
                "rollout_id": f"{method}-{episode}", "status": "completed",
                "success": bool((episode + (method != Method.VANILLA)) % 2),
                "method": method, "config_hash": f"hash-{method}",
                "config_json": {},
                "ms_candidate_u": (_telemetry(episode)
                                   if method != Method.VANILLA else None),
            })
    return pd.DataFrame(rows)


def test_validate_and_flatten_qplanning_heldout_rows():
    arms = validate_qplanning_heldout_cohort(
        _rows(), expected_identities=3, require_complete=True)
    assert set(arms) == set(ARMS)
    boundaries = extract_qplanning_boundaries(arms)
    assert len(boundaries) == 12
    assert set(boundaries.chunk_idx) == {0, 1}
    assert np.allclose(boundaries.tail_minus_prefix_rms, .2)
    assert boundaries.elite_effective_n.between(15.99, 16.01).all()
    features = q_episode_features(boundaries)
    assert len(features) == 6
    assert "negative_q_weighted_first_chunk" in features
    prefix = q_prefix_features(boundaries, max_chunks=3)
    assert len(prefix) == 18


def test_pair_and_gate_keep_every_episode_in_denominator():
    arms = validate_qplanning_heldout_cohort(
        _rows(), expected_identities=3, require_complete=True)
    pair = pair_against_stock(arms, Method.QPLANNING_Q10)
    signal = pair[DIVERSITY_PAIR_KEYS].copy()
    signal["risk"] = [0.1, 0.2, 0.3]
    pair = pair.merge(signal, on=DIVERSITY_PAIR_KEYS, validate="one_to_one")
    sweep = quantile_gate_sweep(
        pair, score_column="risk", grid_size=3, min_selected=1)
    assert sweep.episodes_in_sr_denominator.eq(3).all()
    assert {"high_threshold", "bounded_window"} == set(sweep.gate_kind)
