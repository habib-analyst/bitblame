"""Token-category attribution — §4.2 of METHODOLOGY.md (spec v1.0).

Every per-token KL is tagged; report mean KL per category with bootstrap
CIs (computed from the same Stage-1 forward passes — no extra compute):

  * position bucket: first 25% / middle 50% / last 25% of the sequence —
    diagnoses error accumulation (late >> early => autoregressive
    compounding, the classic quantization failure signature).
  * outlier tokens: top 1% by max-abs activation in the FP16 final hidden
    state — tests whether damage concentrates on outlier activations.
  * domain tag: calibration corpus tag (Phase B: all "prose").

Novel sub-claim (iii): per-category KL with CIs as a built-in forensics
readout.
"""
from __future__ import annotations

import zlib

import numpy as np


def position_bucket_mask(T: int) -> np.ndarray:
    """(T,) int array: 0 = first 25%, 1 = middle 50%, 2 = last 25%."""
    m = np.ones(T, dtype=np.int64)  # middle
    q1, q3 = T // 4, (3 * T) // 4
    m[:q1] = 0
    m[q3:] = 2
    return m


def category_means(
    token_kl: np.ndarray,      # (U, N, T) draw-averaged per-token KL
    outlier_mask: np.ndarray,  # (N, T) bool
) -> dict:
    """Mean KL per (unit, category). Returns dict of (U,) arrays."""
    U, N, T = token_kl.shape
    pos = position_bucket_mask(T)
    out = {}
    for name, sel in (("pos_first25", pos == 0),
                      ("pos_mid50", pos == 1),
                      ("pos_last25", pos == 2)):
        out[name] = token_kl[:, :, sel].mean(axis=(1, 2))
    om = outlier_mask.astype(bool)
    # guard: if a sequence has no flagged tokens, fall back gracefully
    out["outlier_top1pct"] = np.array([
        token_kl[u][om].mean() if om.any() else np.nan for u in range(U)
    ])
    out["non_outlier"] = np.array([
        token_kl[u][~om].mean() if (~om).any() else np.nan for u in range(U)
    ])
    return out


def category_bootstrap_ci(
    token_kl: np.ndarray,       # (U, N, T)
    outlier_mask: np.ndarray,  # (N, T)
    B: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """BCa CIs for per-category mean KL. Resampling unit = sequences:
    category means are recomputed per replicate (§6.1b)."""
    from .bootstrap import bootstrap_mean_ci

    U, N, T = token_kl.shape
    pos = position_bucket_mask(T)
    om = outlier_mask.astype(bool)
    cats = {
        "pos_first25": pos == 0,
        "pos_mid50": pos == 1,
        "pos_last25": pos == 2,
    }
    result: dict = {}
    for u in range(U):
        ucats = {}
        for cname, sel in cats.items():
            # stable per-category seed offset: crc32, NOT hash() — the
            # builtin hash() is salted per process (PYTHONHASHSEED), so
            # hash()-derived seeds made category CIs non-reproducible
            # across runs. Stored Phase-B artifacts are unaffected.
            x = token_kl[u][:, sel].mean(axis=1)  # per-sequence category mean
            ucats[cname] = bootstrap_mean_ci(
                x, B=B, level=level,
                seed=seed + zlib.crc32(cname.encode()) % 997)
        if om.any():
            xo = np.array([token_kl[u, n][om[n]].mean()
                           if om[n].any() else np.nan for n in range(N)])
            xo = xo[~np.isnan(xo)]
            ucats["outlier_top1pct"] = bootstrap_mean_ci(
                xo, B=B, level=level, seed=seed + 11)
        result[u] = ucats
    return result
