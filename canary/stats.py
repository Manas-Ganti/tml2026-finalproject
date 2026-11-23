"""One-sided Mann-Whitney U, Holm-Bonferroni, re-split null, min-bundle-size curve, TPR@1%FPR.

Owner: C · Task: C3 (docs/PLAN.md §5). Pre-registered choices: docs/preregistration.md.
All scores are oriented higher = more member-like.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

BUNDLE_SIZES = (10, 25, 50, 100, 200)
ALPHA = 0.01


def mwu_greater(member, nonmember) -> tuple[float, float]:
    """One-sided MWU (members > nonmembers). Returns (AUC, p). Asymptotic with tie correction."""
    m, n = np.asarray(member, float), np.asarray(nonmember, float)
    if len(m) == 0 or len(n) == 0:
        return float("nan"), float("nan")
    if np.ptp(np.concatenate([m, n])) == 0:            # all scores identical: no evidence at all
        return 0.5, 1.0
    res = _st.mannwhitneyu(m, n, alternative="greater", method="asymptotic")
    return float(res.statistic / (len(m) * len(n))), float(res.pvalue)


def tpr_at_fpr(member, nonmember, fpr: float = 0.01) -> float:
    """TPR at the threshold that lets through at most `fpr` of nonmembers (strictly greater than threshold)."""
    m, n = np.asarray(member, float), np.asarray(nonmember, float)
    thr = np.quantile(n, 1 - fpr, method="higher")
    return float(np.mean(m > thr))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def power_curve(member, nonmember, sizes=BUNDLE_SIZES, reps: int = 500, alpha: float = ALPHA,
                seed: int = 0) -> dict[int, float]:
    """For each bundle size m: fraction of resampled (m members, m nonmembers) bundles with p < alpha."""
    rng = np.random.default_rng(seed)
    m, n = np.asarray(member, float), np.asarray(nonmember, float)
    out = {}
    for k in sizes:
        if k > len(m) or k > len(n):
            continue
        hits = sum(mwu_greater(rng.choice(m, k, replace=False), rng.choice(n, k, replace=False))[1] < alpha
                   for _ in range(reps))
        out[k] = hits / reps
    return out


def min_bundle(power: dict[int, float], target: float = 0.8) -> int | None:
    ok = [k for k, pw in sorted(power.items()) if pw >= target]
    return ok[0] if ok else None


def resplit_null(member, nonmember, reps: int = 200, seed: int = 0) -> np.ndarray:
    """p-values after randomly relabelling the pooled scores (valid null only on a model that saw neither)."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([np.asarray(member, float), np.asarray(nonmember, float)])
    k = len(member)
    ps = []
    for _ in range(reps):
        perm = rng.permutation(pool)
        ps.append(mwu_greater(perm[:k], perm[k:])[1])
    return np.asarray(ps)


def ks_uniform(pvals) -> float:
    """Kolmogorov-Smirnov p-value against Uniform(0, 1)."""
    p = np.asarray(pvals, float)
    p = p[np.isfinite(p)]
    return float(_st.kstest(p, "uniform").pvalue) if len(p) else float("nan")
