"""Paired-bootstrap confidence intervals — §6 of METHODOLOGY.md (spec v1.0).

Resampling unit is ALWAYS the independent unit: calibration sequences for
attribution quantities (a)/(b)/(d), probe items for (c). Tokens within a
sequence are dependent — never resample tokens.

Primary interval estimator: BCa (Efron 1987). Fallback: percentile if the
acceleration constant is numerically unstable (|a_hat| > 0.2 or jackknife
failure); the estimator used is logged per interval.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _ss


def _phi_inv(x: float) -> float:
    return _ss.norm.ppf(np.clip(x, 1e-12, 1 - 1e-12))


def bca_interval(
    theta_hat: float,
    boot: np.ndarray,
    jack: np.ndarray | None,
    level: float = 0.95,
) -> dict:
    """BCa interval from a point estimate, B bootstrap replicates, and
    jackknife replicates (for the acceleration constant)."""
    B = boot.size
    alpha = 1.0 - level
    # bias-correction: fraction of replicates below the estimate
    z0 = _phi_inv((np.sum(boot < theta_hat) + 0.5) / (B + 1.0))
    # acceleration from jackknife
    a_hat, method = 0.0, "percentile"
    if jack is not None and jack.size > 2:
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        if den > 0 and np.isfinite(num / den):
            a_hat = float(num / den)
    if abs(a_hat) <= 0.2:
        method = "bca"
        z_lo, z_hi = _phi_inv(alpha / 2), _phi_inv(1 - alpha / 2)
        a1 = _ss.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
        a2 = _ss.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
        lo = float(np.quantile(boot, np.clip(a1, 0, 1)))
        hi = float(np.quantile(boot, np.clip(a2, 0, 1)))
    else:  # fallback: plain percentile (§6.2)
        lo = float(np.quantile(boot, alpha / 2))
        hi = float(np.quantile(boot, 1 - alpha / 2))
    return {"low": lo, "high": hi, "a_hat": float(a_hat),
            "z0": float(z0), "method": method, "B": int(B)}


def bootstrap_mean_ci(
    x: np.ndarray,
    B: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """BCa CI for the mean of 1-D observations (resampling unit = rows)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)
    # jackknife (leave-one-out means)
    jack = (x.sum() - x) / (n - 1) if n > 2 else None
    out = bca_interval(float(x.mean()), boot, jack, level)
    out["estimate"] = float(x.mean())
    return out


def bootstrap_unit_cis(
    seq_kl: np.ndarray,      # (n_units, n_seqs): per-sequence mean KL
    B: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    top_k: int = 8,
) -> tuple[list[dict], np.ndarray]:
    """Joint bootstrap over units: one shared resample index set per
    replicate, so ranks and P(top-k) come from the same replicates (§6.1d).

    Returns (list of per-unit CI dicts, p_topk array).
    """
    seq_kl = np.asarray(seq_kl, dtype=np.float64)
    U, N = seq_kl.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(B, N))          # (B, N)
    boot_means = seq_kl[:, idx].mean(axis=2)       # (U, B)
    theta = seq_kl.mean(axis=1)                    # (U,)
    # P(unit in top-k) from the same replicates
    ranks = np.argsort(-boot_means, axis=0)        # (U, B) descending
    in_topk = np.zeros((U, B), dtype=bool)
    in_topk[ranks[:top_k, :], np.arange(B)] = True
    p_topk = in_topk.mean(axis=1)
    # per-unit BCa from the marginal replicates + jackknife per unit
    cis = []
    for u in range(U):
        x = seq_kl[u]
        jack = (x.sum() - x) / (N - 1) if N > 2 else None
        ci = bca_interval(float(theta[u]), boot_means[u], jack, level)
        ci["estimate"] = float(theta[u])
        ci["p_topk"] = float(p_topk[u])
        cis.append(ci)
    return cis, p_topk


def split_half_spearman(seq_kl: np.ndarray, seed: int = 0) -> dict:
    """Split-half reliability: Spearman rho between unit rankings on two
    random halves of the sequences (§2.3)."""
    seq_kl = np.asarray(seq_kl, dtype=np.float64)
    N = seq_kl.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    h1, h2 = perm[: N // 2], perm[N // 2 :]
    r1 = _ss.rankdata(-seq_kl[:, h1].mean(axis=1))
    r2 = _ss.rankdata(-seq_kl[:, h2].mean(axis=1))
    rho = float(_ss.spearmanr(r1, r2).statistic)
    return {"rho": rho, "n_half": int(len(h1)), "gate_pass": bool(rho >= 0.9)}


def paired_bootstrap_test(
    delta: np.ndarray,     # per-item paired differences (recipe - base)
    B: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Pre-registered paired test (§6.3, §8.2): BCa CI on mean delta and
    bootstrap p for H0: mean(delta) <= 0."""
    delta = np.asarray(delta, dtype=np.float64).ravel()
    n = delta.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = delta[idx].mean(axis=1)
    theta = float(delta.mean())
    jack = (delta.sum() - delta) / (n - 1) if n > 2 else None
    ci = bca_interval(theta, boot, jack, level)
    p = float((1 + np.sum(boot <= 0)) / (B + 1))
    return {
        "delta_bar": theta,
        "ci_low": ci["low"], "ci_high": ci["high"],
        "ci_method": ci["method"],
        "p_value": p, "B": B, "n": n,
        "verdict": ("SIGNIFICANT_GAIN" if ci["low"] > 0 else
                    "SIGNIFICANT_REGRESSION" if ci["high"] < 0 else
                    "NO_SIGNIFICANT_DIFFERENCE"),
    }


def holm(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    for rank, j in enumerate(order):
        adj[j] = min(1.0, (m - rank) * p[j])
    # enforce monotonicity
    for rank in range(1, m):
        j = order[rank]
        prev = order[rank - 1]
        adj[j] = max(adj[j], adj[prev])
    return adj
