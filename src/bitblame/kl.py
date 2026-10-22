"""Sensitivity metric — §3 of METHODOLOGY.md (spec v1.0).

Per-token directional KL with the FP16 model as reference:

    D_t^(u) = D_KL(p(.|x_<t) || q^(u)(.|x_<t)) = sum_v p_t(v) log(p_t(v)/q_t(v))

Unit blame score:  S_u = mean over sequences and tokens of D_t.
Implementation: accumulate in float64; clip q below at 1e-12; compute
log-softmax in float32 then cast. Plus P50/P95/max tail stats per unit.
"""
from __future__ import annotations

import numpy as np
import torch

Q_FLOOR = 1e-12


def _log_softmax_f32(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.log_softmax(logits.float(), dim=-1)


def token_kl(
    ref_logits: torch.Tensor,
    pert_logits: torch.Tensor,
    chunk: int = 256,
) -> np.ndarray:
    """Per-token D_KL(p_ref || q_pert), returned as float64 numpy (B, T).

    Memory discipline: log-softmax in float32, but the probability tensor
    is NEVER materialized in float64 (that would 8x the transient); the
    per-token sum is accumulated in float64 from fp32 factors. `chunk`
    bounds the transient over (B*T) rows (default 256 rows ~= 50 MB).
    """
    assert ref_logits.shape == pert_logits.shape
    B, T, V = ref_logits.shape
    ref = ref_logits.reshape(-1, V)
    pert = pert_logits.reshape(-1, V)
    n = ref.shape[0]
    out = np.empty(n, dtype=np.float64)
    step = n if not chunk else chunk
    with torch.no_grad():
        for i in range(0, n, step):
            lp = _log_softmax_f32(ref[i : i + step])          # (m, V) fp32
            lq = _log_softmax_f32(pert[i : i + step]).clamp_min(np.log(Q_FLOOR))
            p = lp.exp()                                       # fp32, not fp64
            # sum in float64 from fp32 factors (spec: accumulate in float64)
            kl = (p * (lp - lq)).sum(dim=-1, dtype=torch.float64)
            out[i : i + step] = kl.cpu().numpy()
            del lp, lq, p, kl
    return out.reshape(B, T)


def kl_stats(per_token_kl: np.ndarray) -> dict:
    """Mean + tail stats over a per-token KL array (any shape)."""
    x = np.asarray(per_token_kl, dtype=np.float64).ravel()
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
        "n": int(x.size),
    }
