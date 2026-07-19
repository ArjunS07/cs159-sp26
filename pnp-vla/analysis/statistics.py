"""Statistical primitives used by all analysis sections."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from scipy.stats import binomtest, norm
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    binomtest = norm = average_precision_score = roc_auc_score = None


def wilson_interval(successes: int, n: int, alpha: float = .05) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = 1.959963984540054 if norm is None else float(norm.ppf(1 - alpha / 2))
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def paired_counts(baseline: Iterable[bool], condition: Iterable[bool]) -> dict[str, int]:
    b, c = np.asarray(list(baseline), bool), np.asarray(list(condition), bool)
    if b.shape != c.shape:
        raise ValueError("paired arrays have unequal lengths")
    return {"F_to_S": int((~b & c).sum()), "S_to_F": int((b & ~c).sum()),
            "S_to_S": int((b & c).sum()), "F_to_F": int((~b & ~c).sum())}


def paired_bootstrap_ci(baseline, condition, *, seed: int = 159,
                        n_boot: int = 10_000, alpha: float = .05) -> tuple[float, float]:
    b, c = np.asarray(baseline, float), np.asarray(condition, float)
    if len(b) == 0 or len(b) != len(c):
        return math.nan, math.nan
    delta = c - b
    rng = np.random.default_rng(seed)
    estimates = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))


def discordant_test(f_to_s: int, s_to_f: int) -> float:
    n = f_to_s + s_to_f
    if n == 0:
        return 1.0
    if binomtest is None:  # exact two-sided binomial fallback
        from math import comb
        tail = sum(comb(n, k) for k in range(min(f_to_s, s_to_f) + 1)) / 2 ** n
        return min(1.0, 2 * tail)
    return float(binomtest(f_to_s, n, .5).pvalue)


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def auc_metrics(labels, scores) -> dict[str, float | int]:
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    out = {"n": int(len(y)), "failures": int(y.sum()), "roc_auc": math.nan, "pr_auc": math.nan}
    if len(y) >= 3 and len(np.unique(y)) == 2 and roc_auc_score is not None:
        out.update(roc_auc=float(roc_auc_score(y, s)),
                   pr_auc=float(average_precision_score(y, s)))
    return out


def bootstrap_auc(labels, scores, *, seed: int = 159, n_boot: int = 2000) -> dict[str, float]:
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    base = auc_metrics(y, s)
    rng, roc, pr = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        m = auc_metrics(y[ix], s[ix])
        if np.isfinite(m["roc_auc"]):
            roc.append(m["roc_auc"]); pr.append(m["pr_auc"])
    base.update(roc_ci_low=float(np.quantile(roc, .025)) if roc else math.nan,
                roc_ci_high=float(np.quantile(roc, .975)) if roc else math.nan,
                pr_ci_low=float(np.quantile(pr, .025)) if pr else math.nan,
                pr_ci_high=float(np.quantile(pr, .975)) if pr else math.nan)
    return base
