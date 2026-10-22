"""Mock quantization — §2.1 of METHODOLOGY.md (spec v1.0).

For target bit-width b, tensor w, and quantization group g (256-weight
super-blocks, matching llama.cpp k-quant grouping):

    s_g = max|w_i| / (2^(b-1) - 1)          (symmetric per-group scale)
    w~_i = w_i + eps_i,  eps_i ~ U(-s_g/2, s_g/2)

Moment-matched to real round-to-nearest quantization error (Widrow &
Kollar 2008): same support, E[eps]=0, Var = s_g^2/12.
"""
from __future__ import annotations

import hashlib

import torch


def noise_seed(unit_id: str, draw: int, batch_idx: int, salt: str = "") -> int:
    """Deterministic per-(unit, draw, batch) seed: seed = hash(unit_id, draw, batch)."""
    h = hashlib.sha256(f"{salt}|{unit_id}|{draw}|{batch_idx}".encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def group_scales(w: torch.Tensor, bits: int, group_size: int = 256) -> torch.Tensor:
    """Per-group symmetric scales s_g = max|w_i| / (2^(b-1) - 1).

    Groups are consecutive `group_size` elements of the flattened tensor;
    the tail group is zero-padded implicitly (only real elements matter
    for the max). Returns a 1-D tensor of scales, one per group.
    """
    flat = w.detach().reshape(-1).float()
    n = flat.numel()
    n_groups = (n + group_size - 1) // group_size
    padded = torch.zeros(n_groups * group_size, dtype=torch.float32)
    padded[:n] = flat.abs()
    grouped = padded.reshape(n_groups, group_size)
    amax = grouped.amax(dim=1)
    denom = float(2 ** (bits - 1) - 1)
    return amax / denom


def mock_noise_for_weight(
    w: torch.Tensor,
    bits: int,
    group_size: int,
    seed: int,
) -> torch.Tensor:
    """Draw U(-s_g/2, s_g/2) noise matching w's shape and dtype.

    The noise is generated in float32 and cast to w.dtype, matching the
    precision at which it is applied to the weights.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scales = group_scales(w, bits, group_size)  # (n_groups,)
    n = w.numel()
    # per-element scale lookup
    idx = torch.arange(n, dtype=torch.long) // group_size
    s_elem = scales[idx].clamp_min(0.0)
    u = torch.rand(n, generator=gen, dtype=torch.float32).mul_(2.0).sub_(1.0)
    noise = (u * 0.5 * s_elem).reshape(w.shape)
    return noise.to(dtype=w.dtype)


def apply_mock_noise(
    named_weights: dict[str, torch.Tensor],
    bits: int,
    group_size: int,
    unit_id: str,
    draw: int,
    batch_idx: int,
    salt: str = "",
) -> dict[str, torch.Tensor]:
    """Generate mock-noise tensors for one unit's weights.

    Returns {name: noise}; the caller adds them to the live weights and
    removes them afterwards (or stashes/restores). Noise is redrawn per
    batch per the spec seeding rule.
    """
    seed = noise_seed(unit_id, draw, batch_idx, salt)
    out: dict[str, torch.Tensor] = {}
    for name, w in named_weights.items():
        # deterministic per-name offset (stable across processes/runs)
        name_off = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1000
        out[name] = mock_noise_for_weight(w, bits, group_size, (seed + name_off) % (2 ** 32))
    return out


def quant_step_size(w: torch.Tensor, bits: int) -> float:
    """Fake uniform-quantization step for a whole matrix (prototype parity)."""
    return 2.0 * float(w.detach().abs().max()) / (2 ** bits - 1)
