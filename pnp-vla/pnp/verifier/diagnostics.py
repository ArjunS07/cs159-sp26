"""Zero-simulation diagnostics over already-collected candidate groups.

These are the reusable primitives behind notebook 12 (Phase-0 diagnostics): mode
clustering in executed-prefix space, Bradley-Terry pairwise ranking accuracy,
oracle/majority baselines, and per-stratum aggregation. The notebook stays thin --
it loads candidates, calls these, and keeps only the pre-registered D1 threshold
logic inline. sklearn is optional; without it mode clustering collapses to a single
mode and the mode-level analyses degrade gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # sklearn lives in the [analysis] extra; degrade gracefully without it.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only in minimal installs
    _HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
# Executed-prefix feature space + within-group mode structure
# --------------------------------------------------------------------------- #
def prefix_features(example, prefix_length: int = 10) -> np.ndarray:
    """Flatten a candidate's masked executed prefix into one feature vector."""
    actions = np.asarray(example.actions)[:prefix_length].astype(np.float64)
    mask = np.asarray(example.action_mask)[:prefix_length].astype(bool)
    return np.where(mask[:, None], actions, 0.0).reshape(-1)


def assign_modes(features, *, min_silhouette: float = 0.15) -> np.ndarray:
    """Cluster prefix features into modes; single mode when sklearn/data is thin."""
    n = len(features)
    if n <= 2 or not _HAVE_SKLEARN:
        return np.zeros(n, dtype=int)
    matrix = np.stack(features)
    best_score, best_labels = -1.0, np.zeros(n, dtype=int)
    for k in range(2, min(4, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels if best_score >= min_silhouette else np.zeros(n, dtype=int)


def pairwise_spread(features) -> float:
    """Mean pairwise Euclidean distance between a group's prefix features."""
    matrix = np.stack(features)
    if len(matrix) < 2:
        return 0.0
    distances = np.sqrt(((matrix[:, None] - matrix[None, :]) ** 2).sum(-1))
    upper = np.triu_indices(len(matrix), 1)
    return float(distances[upper].mean())


def add_mode_structure(cand: pd.DataFrame, *, prefix_col: str = "prefix") -> pd.DataFrame:
    """Attach per-group ``mode``, ``n_modes``, and ``spread_tercile`` columns in place."""
    cand["mode"] = -1
    spread: dict = {}
    for gid, index in cand.groupby("group_id").groups.items():
        features = list(cand.loc[index, prefix_col])
        cand.loc[index, "mode"] = assign_modes(features)
        spread[gid] = pairwise_spread(features)
    cand["n_modes"] = cand.groupby("group_id")["mode"].transform("nunique")
    group_spread = pd.Series(spread)
    low_q, high_q = group_spread.quantile([1 / 3, 2 / 3])
    cand["spread_tercile"] = cand.group_id.map(
        lambda g: "low" if spread[g] <= low_q else ("mid" if spread[g] <= high_q else "high"))
    return cand


# --------------------------------------------------------------------------- #
# Group-level ranking / outcome metrics
# --------------------------------------------------------------------------- #
def group_pair_accuracy(scores, successes) -> float:
    """Bradley-Terry pairwise accuracy: P(score(success) > score(failure)) in one group."""
    scores = np.asarray(scores, dtype=float)
    successes = np.asarray(successes, dtype=int)
    if np.isnan(scores).any():
        return float("nan")
    positive, negative = scores[successes == 1], scores[successes == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0) + 0.5 * (differences == 0)).mean())


def default_success(group: pd.DataFrame) -> int:
    """Outcome of the group's ``default`` candidate (fallback: first row)."""
    marked = group[group.is_default]
    return int(marked.success.iloc[0]) if len(marked) else int(group.success.iloc[0])


def group_oracle_uplift(group: pd.DataFrame) -> float:
    """Success headroom of a perfect selector over the default candidate."""
    return float(group.success.max() - default_success(group))


def mode_ranking_accuracy(group: pd.DataFrame) -> float:
    """Does the model order the group's modes by their empirical success rate?"""
    if np.isnan(group["score"]).any():
        return float("nan")
    modes = group.groupby("mode").agg(
        score=("score", "mean"), success_rate=("success", "mean")).reset_index()
    if len(modes) < 2:
        return float("nan")
    pairs = [
        float((modes.score[i] > modes.score[j]) == (modes.success_rate[i] > modes.success_rate[j]))
        for i in range(len(modes)) for j in range(i + 1, len(modes))
        if modes.success_rate[i] != modes.success_rate[j]
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


def within_mode_pair_fraction(group: pd.DataFrame) -> float:
    """Fraction of (success x failure) ranking pairs that fall inside a single mode."""
    positive, negative = group[group.success == 1], group[group.success == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    total = len(positive) * len(negative)
    within = sum(int(p == n) for p in positive["mode"] for n in negative["mode"])
    return within / total


def majority_mode_success(group: pd.DataFrame) -> float:
    """Expected success of a training-free selector that executes the largest mode."""
    top = group.groupby("mode").size().idxmax()
    return float(group[group["mode"] == top]["success"].mean())


# --------------------------------------------------------------------------- #
# Per-candidate table + stratified aggregation
# --------------------------------------------------------------------------- #
def build_candidate_table(examples, model=None, device=None, *, prefix_length: int = 10,
                          config=None, config_zc=None, progress=None) -> pd.DataFrame:
    """One row per candidate: conditioned score, action-only control score, outcome, prefix.

    ``model=None`` yields NaN scores so the model-free diagnostics still run.
    """
    import torch

    from .train import AdvantageTrainConfig, group_examples, score_group

    config = config or AdvantageTrainConfig(prefix_length=prefix_length, zero_context=False)
    config_zc = config_zc or AdvantageTrainConfig(prefix_length=prefix_length, zero_context=True)
    groups = group_examples(examples)
    items = groups.items()
    if progress is not None:
        items = progress(items, total=len(groups))
    rows = []
    for gid, members in items:
        if model is not None:
            with torch.no_grad():
                _, advantage, _ = score_group(model, members, device, config)
                _, advantage_zc, _ = score_group(model, members, device, config_zc)
            scores = advantage.detach().cpu().numpy().reshape(-1)
            scores_zc = advantage_zc.detach().cpu().numpy().reshape(-1)
        else:
            scores = scores_zc = [float("nan")] * len(members)
        for j, example in enumerate(members):
            rows.append({
                "group_id": gid, "benchmark": example.benchmark,
                "stratum": example.uncertainty_stratum or "unknown",
                "kind": example.candidate_kind,
                "is_default": example.candidate_kind == "default",
                "score": float(scores[j]), "score_zc": float(scores_zc[j]),
                "success": int(example.success),
                "prefix": prefix_features(example, prefix_length),
                "return_target": example.return_target,
            })
    return pd.DataFrame(rows)


def stratified(cand: pd.DataFrame, by: str, *, score_col: str = "score",
               seed: int = 42, n_bootstrap: int = 10_000) -> pd.DataFrame:
    """Ranking / control-gap / oracle summary within each level of ``by``."""
    from .train import bootstrap_interval

    out = []
    for key, subset in cand.groupby(by):
        pair, pair_zc, oracle = [], [], []
        for _, group in subset.groupby("group_id"):
            accuracy = group_pair_accuracy(group[score_col], group["success"])
            if not np.isnan(accuracy):
                pair.append(accuracy)
                accuracy_zc = group_pair_accuracy(group["score_zc"], group["success"])
                if not np.isnan(accuracy_zc):
                    pair_zc.append(accuracy_zc)
            oracle.append(group_oracle_uplift(group))
        ci = bootstrap_interval(pair, seed, n_bootstrap) if pair else [float("nan"), float("nan")]
        out.append({
            by: key, "groups": subset.group_id.nunique(), "discordant": len(pair),
            "ranking": np.mean(pair) if pair else float("nan"),
            "ci_lo": ci[0], "ci_hi": ci[1],
            "action_only": np.mean(pair_zc) if pair_zc else float("nan"),
            "control_gap": (np.mean(pair) - np.mean(pair_zc)) if pair and pair_zc else float("nan"),
            "oracle_uplift": np.mean(oracle) if oracle else float("nan"),
        })
    return pd.DataFrame(out)


def deployment_stratum_stats(cand: pd.DataFrame, *, stratum_col: str = "spread_tercile",
                             stratum_value: str = "high", seed: int = 42,
                             n_bootstrap: int = 10_000) -> dict:
    """Compute the raw numbers the pre-registered D1 gate thresholds are applied to.

    Threshold/decision logic stays inline in the notebook; this only produces the
    ranking, bootstrap CI, control gap, oracle uplift, and discordant-group count on
    the deployment-relevant stratum.
    """
    from .train import bootstrap_interval

    deployment = cand[cand[stratum_col] == stratum_value]
    grouped = deployment.groupby("group_id")
    pair = [a for _, g in grouped
            if not np.isnan(a := group_pair_accuracy(g["score"], g["success"]))]
    pair_zc = [a for _, g in grouped
               if not np.isnan(a := group_pair_accuracy(g["score_zc"], g["success"]))]
    ci = bootstrap_interval(pair, seed, n_bootstrap) if pair else [float("nan"), float("nan")]
    oracle = ([group_oracle_uplift(g) for _, g in grouped] if len(deployment) else [])
    return {
        "ranking": float(np.mean(pair)) if pair else float("nan"),
        "ci": [float(ci[0]), float(ci[1])],
        "control_gap": (float(np.mean(pair)) - float(np.mean(pair_zc)))
        if pair and pair_zc else float("nan"),
        "oracle_uplift": float(np.mean(oracle)) if oracle else float("nan"),
        "discordant_n": len(pair),
    }


# --------------------------------------------------------------------------- #
# Checkpoint reconstruction
# --------------------------------------------------------------------------- #
def build_verifier(arch: dict, state_dict, device=None):
    """Reconstruct a :class:`CompactAdvantageVerifier` from its registered metadata."""
    from .model import CompactAdvantageVerifier

    model = CompactAdvantageVerifier(
        action_width=int(arch.get("action_width", 64)),
        dropout=float(arch.get("dropout", 0.2)),
        conditioning=arch.get("architecture", "multiplicative"))
    model.load_state_dict(state_dict)
    if device is not None:
        model = model.to(device)
    return model.eval()
