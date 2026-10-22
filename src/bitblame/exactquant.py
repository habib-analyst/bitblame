"""Stage 2 — exact top-k verification + slope calibration (§2.4, §7.2).

For the top-k units by Stage-1 S_u, replace the unit's weights with their
*exactly quantized-then-dequantized* values at b in {3, 4, 6}, forward
pass, measure output KL (§3). This is exact w.r.t. the deployed artifact
(up to the torch reimplementation of the quantizer).

Phase-B scope: deterministic per-group symmetric round-to-nearest INT
quantization (group = 256, matching the mock screen's grouping). Faithful
k-quant super-block math (sub-block scales, importance weighting) is
verified against llama.cpp in the closed loop (Stage 5) — until the
llama.cpp build lands, this module is the exact-math verification step
and the report states that boundary explicitly.

Slope calibration (§7.2): fit log l_u(b) = log c_u - alpha_u (b - b_ref)
by least squares on the 3 measured points, then shrink toward theory:

    alpha_u = lambda * alpha_hat + (1 - lambda) * alpha_theory,
    lambda = R^2_fit   (R^2 < 0.5 -> lambda = 0; never extrapolate below b=3)
"""
from __future__ import annotations

import numpy as np
import torch

from . import kl as klmod
from .modeling import _resolve, forward_logits


def exact_quantize_weight(
    w: torch.Tensor, bits: int, group_size: int = 256
) -> torch.Tensor:
    """Deterministic symmetric per-group round-to-nearest INT quantize +
    dequantize. Returns the dequantized weight (same shape/dtype as w)."""
    shape, dtype = w.shape, w.dtype
    flat = w.detach().reshape(-1).float()
    n = flat.numel()
    n_groups = (n + group_size - 1) // group_size
    padded = torch.zeros(n_groups * group_size, dtype=torch.float32)
    padded[:n] = flat
    grouped = padded.reshape(n_groups, group_size)
    amax = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    denom = float(2 ** (bits - 1) - 1)
    s = amax / denom
    q = torch.round(grouped / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    deq = (q * s).reshape(-1)[:n].reshape(shape)
    return deq.to(dtype)


def measure_exact_kl(
    model: torch.nn.Module,
    unit,                      # Unit
    ref_logits_list: list[torch.Tensor],  # per-batch cached ref logits
    batches: list[torch.Tensor],
    bits_list: tuple = (3, 4, 6),
    group_size: int = 256,
) -> dict[int, float]:
    """Swap in exactly-quantized weights for `unit`, measure mean-token KL
    per bit-width. Model weights are restored afterwards."""
    wnames = unit.weight_names
    stash = {n: _resolve(model, n).detach().clone() for n in wnames}
    out: dict[int, float] = {}
    try:
        for b in bits_list:
            with torch.no_grad():
                for n in wnames:
                    _resolve(model, n).copy_(
                        exact_quantize_weight(stash[n], b, group_size))
            kls = []
            with torch.no_grad():
                for batch, ref_logits in zip(batches, ref_logits_list):
                    pert = forward_logits(model, batch)
                    kls.append(klmod.token_kl(ref_logits, pert).mean())
            out[b] = float(np.mean(kls))
    finally:
        with torch.no_grad():
            for n, orig in stash.items():
                _resolve(model, n).copy_(orig)
    return out


def fit_slope(
    kl_by_bits: dict[int, float],
    b_ref: int = 4,
    alpha_theory: float = 1.3862943611198906,
) -> dict:
    """Least-squares fit of log KL vs bit-width + shrinkage toward theory."""
    bs = np.array(sorted(kl_by_bits), dtype=np.float64)
    ys = np.array([kl_by_bits[b] for b in bs], dtype=np.float64)
    ys = np.clip(ys, 1e-300, None)
    ly = np.log(ys)
    x = bs - b_ref
    # least squares: ly = c0 - alpha * x
    A = np.stack([np.ones_like(x), -x], axis=1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    c0, alpha_hat = coef
    pred = c0 - alpha_hat * x
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lam = r2 if r2 >= 0.5 else 0.0
    alpha = lam * alpha_hat + (1 - lam) * alpha_theory
    return {
        "alpha_hat": float(alpha_hat),
        "alpha": float(alpha),
        "r2": float(r2),
        "lambda": float(lam),
        "log_c": float(c0),
        "kl_by_bits": {int(k): float(v) for k, v in kl_by_bits.items()},
    }
