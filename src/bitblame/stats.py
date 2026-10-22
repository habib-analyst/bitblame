"""Statistical utilities for BitBlame — public API.

Thin wrappers over the implemented machinery (METHODOLOGY.md §6):

- ``paired_bootstrap_bca`` — pre-registered paired test on per-item
  deltas (``bootstrap.paired_bootstrap_test``).
- ``null_envelope`` — null-floor distribution of sensitivity scores
  (``bootstrap.bootstrap_unit_cis`` on a null scan).
- ``fdr_control`` — Benjamini–Hochberg FDR-adjusted p-values (this
  module; the Holm FWER procedure lives in ``bootstrap.holm`` and is
  what §6.3 prescribes for top-k exact-stage claims).
"""
from __future__ import annotations

import numpy as np

from . import bootstrap as _bs


def paired_bootstrap_bca(delta, B: int = 10000, level: float = 0.95,
                         seed: int = 0) -> dict:
    """BCa CI + bootstrap p for H0: mean(delta) <= 0 on paired deltas.

    ``delta``: per-item paired differences (e.g. recipe − baseline
    scores). Returns ``{delta_bar, ci_low, ci_high, ci_method, p_value,
    B, n, verdict}`` with verdict in ``SIGNIFICANT_GAIN /
    NO_SIGNIFICANT_DIFFERENCE / SIGNIFICANT_REGRESSION``.
    """
    return _bs.paired_bootstrap_test(np.asarray(delta, dtype=np.float64),
                                     B=B, level=level, seed=seed)


def null_envelope(null_seq_kl, B: int = 2000, level: float = 0.95,
                  seed: int = 0) -> dict:
    """Null-floor envelope from a null (b=8 mock-noise) scan.

    ``null_seq_kl``: (n_units, n_seqs) per-sequence mean KL from the
    null run. Returns per-unit CIs plus ``floor`` — the
    ``max(0, max CI_high)`` threshold the significance gate (§7.1)
    compares each unit's CI_low against.
    """
    cis, _ = _bs.bootstrap_unit_cis(
        np.asarray(null_seq_kl, dtype=np.float64),
        B=B, level=level, seed=seed)
    floor = max(0.0, max(c["high"] for c in cis)) if cis else 0.0
    return {"per_unit": cis, "floor": float(floor),
            "B": int(B), "level": float(level)}


def fdr_control(pvals, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values.

    Returns adjusted p-values in the input order; reject where
    ``adjusted <= alpha``. (For the §6.3 top-k claims the spec
    prescribes Holm FWER — see ``bootstrap.holm`` — not FDR.)
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p-value down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out
