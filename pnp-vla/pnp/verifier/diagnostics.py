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
                "chunk_position": float(example.chunk_position),
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
# Per-denoising-step uncertainty reconstruction
# --------------------------------------------------------------------------- #
# The probe runs at every euler step; noise level s decreases 0.9 -> 0.1. The stored
# `uncertainty_stratum` is a tercile of the trajectory MEAN of these per-step u_mean
# values (see collection.build_stratified_manifest). These helpers recover the raw
# per-s signal so a gate can be defined at a chosen noise level instead of the average.
_U_STEP_LEVELS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _u_step_column(s) -> str:
    """Stable column name for a noise level, e.g. 0.5 -> 'u_s50'."""
    return f"u_s{int(round(float(s) * 100)):02d}"


def _pivot_euler_rows(rows) -> dict:
    """(rollout_id, chunk_idx) -> {u_s<level>: mean u_mean} from pnp_euler_steps rows.

    Pure and offline (no store access) so it is unit-testable. Repeated rows at the
    same (key, noise level) are averaged.
    """
    from collections import defaultdict

    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["rollout_id"], int(row["chunk_idx"]))
        acc[key][_u_step_column(row["s"])].append(float(row["u_mean"]))
    return {key: {col: float(np.mean(vals)) for col, vals in cols.items()}
            for key, cols in acc.items()}


def attach_per_step_uncertainty(cand: pd.DataFrame, development, store, *,
                                source_experiments=None, batch: int = 200,
                                progress=None) -> pd.DataFrame:
    """Attach per-denoising-step U columns to ``cand``, keyed by candidate group.

    The stored ``uncertainty_stratum`` collapses the whole denoising trajectory into a
    single tercile bucket. This reconstructs, per candidate group, the source
    ``(rollout_id, chunk_idx)`` it was branched from and pulls that chunk's per-euler-step
    ``u_mean`` from ``pnp_euler_steps``, exposing U at each noise level.

    Adds columns ``u_s90..u_s10`` (percent noise level; ``u_s10`` is the endpoint, nearest
    data), plus ``u_traj_mean`` (mean over available steps, the stratum's basis),
    ``u_endpoint`` (== ``u_s10``), and ``u_mid`` (nearest s=0.5). Groups whose source chunk
    is not found keep NaN. Modifies and returns ``cand``.

    ``development`` is the ``CleanChunkExample`` list (it carries the identity fields the
    scored table drops). ``source_experiments`` optionally narrows the ``rollouts`` lookup
    to the base experiments that own the probe rows -- faster and less ambiguous.
    """
    from collections import defaultdict

    # 1. candidate_group_id -> source identity (examples carry these fields; cand drops them).
    identity = {}
    for example in development:
        gid = example.candidate_group_id
        if gid and gid not in identity:
            identity[gid] = (example.benchmark, example.suite, int(example.task_idx),
                             int(example.episode_idx), int(example.chunk_idx))
    wanted4 = {ident[:4] for ident in identity.values()}

    # 2. (benchmark, suite, task_idx, episode_idx) -> [source rollout_id] from `rollouts`.
    def _configure_rollouts(query):
        query = query.select("rollout_id,benchmark,suite,task_idx,episode_idx")
        return query.in_("experiment", list(source_experiments)) if source_experiments else query

    ident_to_rids = defaultdict(list)
    for row in store.fetch_all("rollouts", configure=_configure_rollouts):
        key = (row["benchmark"], row["suite"], int(row["task_idx"]), int(row["episode_idx"]))
        if key in wanted4:
            ident_to_rids[key].append(row["rollout_id"])

    # 3. pull pnp_euler_steps for those source rollouts, batched over rollout_id.
    rids = sorted({rid for group in ident_to_rids.values() for rid in group})
    starts = range(0, len(rids), batch)
    starts = progress(starts) if progress else starts
    euler_rows = []
    for start in starts:
        ids = rids[start:start + batch]
        euler_rows.extend(store.fetch_all(
            "pnp_euler_steps", configure=lambda query, ids=ids: query.select(
                "rollout_id,chunk_idx,euler_step,s,u_mean").in_("rollout_id", ids)))
    per_chunk = _pivot_euler_rows(euler_rows)

    # 4. per group, pick the source rollout that actually has probe rows at its chunk.
    group_u = {}
    for gid, (bench, suite, task, episode, chunk) in identity.items():
        for rid in ident_to_rids.get((bench, suite, task, episode), []):
            cols = per_chunk.get((rid, chunk))
            if cols:
                group_u[gid] = cols
                break

    # 5. write columns onto cand.
    step_cols = [_u_step_column(s) for s in _U_STEP_LEVELS]
    for col in step_cols:
        cand[col] = cand["group_id"].map(lambda g, c=col: group_u.get(g, {}).get(c, np.nan))
    present = [c for c in step_cols if cand[c].notna().any()]
    cand["u_traj_mean"] = cand[present].mean(axis=1) if present else np.nan
    cand["u_endpoint"] = cand[_u_step_column(min(_U_STEP_LEVELS))]
    cand["u_mid"] = cand[_u_step_column(min(_U_STEP_LEVELS, key=lambda s: abs(s - 0.5)))]
    matched = int(cand["group_id"].isin(group_u).sum())
    print("attached per-step U: %d/%d candidate rows, %d/%d groups matched"
          % (matched, len(cand), len(group_u), cand["group_id"].nunique()))
    return cand


def stratify_by_u_level(cand: pd.DataFrame, u_col: str, *, n_buckets: int = 3,
                        labels=("low", "mid", "high")) -> pd.DataFrame:
    """Re-run the 0a stratified summary using a chosen per-step U column as the stratum.

    Buckets groups into terciles of ``u_col`` (group-level value) and calls ``stratified``.
    Lets you compare, e.g., a gate on endpoint U (``u_s10``) vs mid-trajectory U (``u_s50``)
    vs the stored trajectory-average stratum.
    """
    group_u = cand.groupby("group_id")[u_col].first()
    edges = np.array(group_u.quantile(np.linspace(0, 1, n_buckets + 1).tolist()), dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = pd.cut(group_u, bins=np.unique(edges), labels=list(labels)[:len(np.unique(edges)) - 1])
    col = f"{u_col}_bucket"
    cand[col] = cand["group_id"].map(bucket.to_dict())
    return stratified(cand[cand[col].notna()], col)


# --------------------------------------------------------------------------- #
# Wave-0 model-free screening (no verifier scores; runs before any model is trusted)
# --------------------------------------------------------------------------- #
def prefix_distance_disagreement(cand: pd.DataFrame, *, n_bins: int = 8) -> tuple:
    """Label disagreement vs prefix distance within groups (the H-chaos probe).

    For every unordered candidate pair inside a group, record the Euclidean distance
    between their executed-prefix vectors and whether their success labels differ. Pairs
    are pooled across groups and binned by distance (equal-mass quantile bins). A curve of
    P(labels differ) that RISES with distance means outcomes vary smoothly with the prefix
    (a learnable causal structure); a FLAT curve means disagreement is independent of the
    prefix, i.e. within-group labels are coin flips (chaos dominates).

    Returns ``(table, slope)`` where ``table`` has ``{distance_bin, dist_mid, p_disagree,
    n}`` and ``slope`` is the least-squares slope of ``p_disagree`` vs ``dist_mid`` (>0
    rising, ~0 flat). Model-free: uses only stored prefixes and success labels.
    """
    distances, disagree = [], []
    for _, group in cand.groupby("group_id"):
        features = np.stack([np.asarray(p, dtype=float) for p in group["prefix"]])
        success = group["success"].to_numpy()
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(float(np.linalg.norm(features[i] - features[j])))
                disagree.append(int(success[i] != success[j]))
    columns = ["distance_bin", "dist_mid", "p_disagree", "n"]
    if not distances:
        return pd.DataFrame(columns=columns), float("nan")
    distances = np.asarray(distances)
    disagree = np.asarray(disagree)
    edges = np.unique(np.quantile(distances, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(distances, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.any():
            rows.append({"distance_bin": b, "dist_mid": float(distances[mask].mean()),
                         "p_disagree": float(disagree[mask].mean()), "n": int(mask.sum())})
    table = pd.DataFrame(rows, columns=columns)
    slope = (float(np.polyfit(table["dist_mid"], table["p_disagree"], 1)[0])
             if len(table) >= 2 else float("nan"))
    return table, slope


_HANDCRAFTED_FEATURE_NAMES = ("jerk", "smoothness", "grip_mean", "grip_std",
                              "grip_switches", "dist_centroid", "dist_default",
                              "mode_size", "chunk_position")


def _candidate_geoms(prefix_vec, prefix_length: int):
    """Shape features of one executed prefix: jerk, smoothness, gripper-channel stats."""
    actions = np.asarray(prefix_vec, dtype=float).reshape(prefix_length, -1)
    diffs = np.diff(actions, axis=0)
    jerk = float(np.linalg.norm(np.diff(diffs, axis=0))) if len(diffs) > 1 else 0.0
    smoothness = float(np.linalg.norm(diffs))
    grip = actions[:, -1]
    grip_switches = float((np.abs(np.diff(np.sign(grip))) > 0).sum())
    return jerk, smoothness, float(grip.mean()), float(grip.std()), grip_switches


def handcrafted_features(cand: pd.DataFrame, *, prefix_length: int = 10) -> pd.DataFrame:
    """Hand-made geometric features per candidate for the boring baseline (no learning).

    Per-candidate shape features (jerk, smoothness, gripper-channel mean/std/switches) plus
    group-relative geometry (distance to the group centroid and to the default candidate),
    the candidate's mode size, and its chunk_position. Returns a DataFrame aligned to
    ``cand.index`` with the columns in ``_HANDCRAFTED_FEATURE_NAMES``.
    """
    records = {}
    for _, group in cand.groupby("group_id"):
        prefixes = {idx: np.asarray(group.loc[idx, "prefix"], dtype=float) for idx in group.index}
        centroid = np.mean(list(prefixes.values()), axis=0)
        default_rows = group.index[group["is_default"]]
        default_vec = prefixes[default_rows[0]] if len(default_rows) else centroid
        mode_size = (group.groupby("mode")["mode"].transform("size")
                     if "mode" in group else pd.Series(len(group), index=group.index))
        for idx in group.index:
            geoms = _candidate_geoms(prefixes[idx], prefix_length)
            records[idx] = [*geoms,
                            float(np.linalg.norm(prefixes[idx] - centroid)),
                            float(np.linalg.norm(prefixes[idx] - default_vec)),
                            float(mode_size.loc[idx]),
                            float(group.loc[idx].get("chunk_position", 0.0))]
    return pd.DataFrame.from_dict(
        records, orient="index", columns=list(_HANDCRAFTED_FEATURE_NAMES)).reindex(cand.index)


def fit_gbt_baseline(cand: pd.DataFrame, *, prefix_length: int = 10, n_splits: int = 5,
                     seed: int = 0) -> dict:
    """Gradient-boosted-tree boring baseline on handcrafted features (H-representation test).

    Predicts candidate success under GroupKFold on ``group_id`` (no candidate leakage across
    folds), collects out-of-fold probabilities, and scores them with the same
    ``group_pair_accuracy`` used for the deep model -- so the numbers are directly comparable.
    GBT ~ deep model => the deep model is not the bottleneck; GBT wins => representation is;
    GBT ~ chance => strong H-chaos evidence. Requires sklearn ([analysis] extra).
    """
    if not _HAVE_SKLEARN:
        return {"ranking": float("nan"), "n_groups": 0, "importances": {}, "note": "sklearn missing"}
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    features = handcrafted_features(cand, prefix_length=prefix_length)
    matrix = features.to_numpy(dtype=float)
    success = cand["success"].to_numpy()
    groups = cand["group_id"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return {"ranking": float("nan"), "n_groups": int(cand.group_id.nunique()),
                "importances": {}, "note": "need >= 2 groups"}
    oof = np.full(len(cand), np.nan)
    importances = np.zeros(matrix.shape[1])
    folds = 0
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(matrix, success, groups):
        if len(np.unique(success[train_idx])) < 2:  # single-class fold -> unusable
            continue
        model = GradientBoostingClassifier(random_state=seed)
        model.fit(matrix[train_idx], success[train_idx])
        oof[test_idx] = model.predict_proba(matrix[test_idx])[:, 1]
        importances += model.feature_importances_
        folds += 1
    scored = cand.assign(_gbt=oof)
    pair = [a for _, group in scored.groupby("group_id")
            if not np.isnan(a := group_pair_accuracy(group["_gbt"], group["success"]))]
    return {
        "ranking": float(np.mean(pair)) if pair else float("nan"),
        "n_groups": len(pair),
        "importances": dict(sorted(
            zip(_HANDCRAFTED_FEATURE_NAMES, importances / folds if folds else importances),
            key=lambda kv: -kv[1])),
    }


# --- training-free and score-based selectors, scored as deployed success --- #
def _mode_medoid(group: pd.DataFrame, mode):
    """Row index of the candidate closest to its mode's prefix centroid."""
    subset = group[group["mode"] == mode]
    features = np.stack([np.asarray(p, dtype=float) for p in subset["prefix"]])
    centroid = features.mean(axis=0)
    return subset.index[int(np.argmin(((features - centroid) ** 2).sum(axis=1)))]


def _select_random(group: pd.DataFrame) -> float:
    return float(group["success"].mean())


def _select_max_dist_default(group: pd.DataFrame) -> float:
    prefixes = {idx: np.asarray(group.loc[idx, "prefix"], dtype=float) for idx in group.index}
    default_rows = group.index[group["is_default"]]
    reference = (prefixes[default_rows[0]] if len(default_rows)
                 else np.mean(list(prefixes.values()), axis=0))
    pick = max(group.index, key=lambda i: np.linalg.norm(prefixes[i] - reference))
    return float(group.loc[pick, "success"])


def _select_largest_mode_medoid(group: pd.DataFrame) -> float:
    largest = group.groupby("mode").size().idxmax()
    return float(group.loc[_mode_medoid(group, largest), "success"])


def _select_pick_best(group: pd.DataFrame, score_col: str) -> float:
    scores = group[score_col]
    if scores.isna().any():
        return float("nan")
    return float(group.loc[scores.idxmax(), "success"])


def _select_reject_worst(group: pd.DataFrame, score_col: str, quantile: float) -> float:
    scores = group[score_col]
    if scores.isna().any():
        return float("nan")
    survivors = group[scores >= scores.quantile(quantile)]
    return float(survivors["success"].mean()) if len(survivors) else float(group["success"].mean())


def _select_best_mode_medoid(group: pd.DataFrame, score_col: str) -> float:
    scores = group[score_col]
    if scores.isna().any():
        return float("nan")
    best = group.groupby("mode").apply(lambda g: g[score_col].mean()).idxmax()
    return float(group.loc[_mode_medoid(group, best), "success"])


def selector_zoo(cand: pd.DataFrame, *, score_cols=("score",), quantile: float = 0.25,
                 seed: int = 42, n_bootstrap: int = 10_000) -> pd.DataFrame:
    """Deployed success of training-free and score-based selectors vs the default (0.650 bar).

    Each selector maps a candidate group to the success of the candidate it would execute (or
    an expected success for stochastic selectors). Aggregated across groups into deployed
    success + uplift over the default candidate, with a bootstrap CI. Score-based selectors
    are evaluated for BOTH operators -- pick-best (argmax score) and reject-worst (drop the
    bottom ``quantile`` by score, then a uniform pick from survivors) -- for every column in
    ``score_cols``, so any scorer (GBT, current verifier, a future rebuild) can be compared.
    Requires ``mode`` (call ``add_mode_structure`` first) for the mode-based selectors.
    """
    from .train import bootstrap_interval

    groups = list(cand.groupby("group_id"))
    default_mean = float(np.mean([default_success(g) for _, g in groups]))
    rows = []

    def _record(name, score_col, selector):
        values = [v for _, group in groups if not np.isnan(v := selector(group))]
        if not values:
            return
        ci = bootstrap_interval(values, seed, n_bootstrap)
        rows.append({"selector": name, "score_col": score_col or "-",
                     "deployed_success": float(np.mean(values)),
                     "uplift_vs_default": float(np.mean(values)) - default_mean,
                     "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "n_groups": len(values)})

    _record("random", None, _select_random)
    _record("majority_mode", None, majority_mode_success)
    _record("max_dist_default", None, _select_max_dist_default)
    _record("largest_mode_medoid", None, _select_largest_mode_medoid)
    for col in score_cols:
        _record("pick_best", col, lambda g, c=col: _select_pick_best(g, c))
        _record("reject_worst", col, lambda g, c=col: _select_reject_worst(g, c, quantile))
        _record("best_mode_medoid", col, lambda g, c=col: _select_best_mode_medoid(g, c))
    result = pd.DataFrame(rows)
    result.attrs["default_success"] = default_mean
    return result


# --- selector-quality -> deployed-uplift simulation (sets the target ranking) --- #
def _best_of_k(m: int, s: int, k: int) -> float:
    """Exact best-of-k oracle success: probability that k of m candidates include a success."""
    from math import comb

    if k >= m:
        return 1.0 if s > 0 else 0.0
    return 1.0 - comb(m - s, k) / comb(m, k)


def simulate_selector_uplift(cand: pd.DataFrame, *,
                             r_grid=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0),
                             k_grid=(1, 2, 4, 8), group_col: str = "group_id") -> pd.DataFrame:
    """Deployed success as a function of selector ranking accuracy ``r`` and budget ``k``.

    Models a selector of pairwise ranking accuracy ``r`` as a noisy oracle: with probability
    ``q = max(0, 2r - 1)`` it returns the best of ``k`` sampled candidates (exact combinatorial
    oracle), otherwise a uniformly random one (group mean). Averaged over groups with at least
    ``k`` candidates. Labels only -- no model. Converts "ranking above 0.5" into "the minimum
    ``r`` needed to beat the 0.650 default at a given ``k``".
    """
    groups = [(int(group["success"].sum()), len(group))
              for _, group in cand.groupby(group_col)]
    default_mean = float(np.mean([default_success(group)
                                  for _, group in cand.groupby(group_col)]))
    rows = []
    for k in k_grid:
        eligible = [(s, m) for s, m in groups if m >= k]
        if not eligible:
            continue
        best = np.array([_best_of_k(m, s, k) for s, m in eligible])
        random_pick = np.array([s / m for s, m in eligible])
        for r in r_grid:
            q = max(0.0, 2 * r - 1)
            success = float(np.mean(q * best + (1 - q) * random_pick))
            rows.append({"k": k, "r": r, "deployed_success": success,
                         "uplift_vs_default": success - default_mean, "n_groups": len(eligible)})
    return pd.DataFrame(rows)


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
