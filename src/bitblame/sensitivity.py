"""Per-layer quantization sensitivity measurement — public API.

Thin wrappers over the implemented pipeline (METHODOLOGY.md):

- ``rank_layers`` — Stage-1 LOLO mock-noise scan (``lolo.py`` +
  ``kl.py`` + ``mockquant.py``), returning per-unit blame scores.
- ``sensitivity_report`` — scan + null floor + BCa CIs + significance
  gate + split-half reliability (``bootstrap.py``), returning the
  analysis dict the optimizer consumes.
"""
from __future__ import annotations

import numpy as np
import torch

from . import bootstrap as _bs
from .config import ScanConfig
from .lolo import run_lolo_scan
from .modeling import list_units


def _as_batches(calibration, batch_size: int) -> list[torch.Tensor]:
    """Accept a (B, T) tensor, a list of (B, T) tensors, or a list of
    1-D sequences; return a list of (B, T) batches."""
    if isinstance(calibration, torch.Tensor):
        if calibration.dim() == 1:
            return [calibration.unsqueeze(0)]
        return [calibration[i:i + batch_size]
                for i in range(0, calibration.shape[0], batch_size)]
    batches: list[torch.Tensor] = []
    for item in calibration:
        t = item if isinstance(item, torch.Tensor) else torch.tensor(item)
        if t.dim() == 1:
            batches.append(t.unsqueeze(0))
        else:
            batches.extend(t[i:i + batch_size]
                           for i in range(0, t.shape[0], batch_size))
    if not batches:
        raise ValueError("empty calibration input")
    return batches


def _n_layers(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "config_n_layers"):
        return int(model.config_n_layers)
    raise ValueError("cannot infer n_layers: pass units explicitly")


def rank_layers(model, calibration, bits: int = 4, n_draws: int = 2,
                granularity: str = "block", seed: int = 20261014,
                batch_size: int = 2, units=None) -> list[dict]:
    """Per-unit blame scores under mock-noise perturbation (Stage 1).

    ``calibration``: (B, T) token-id tensor or list of batches/sequences.
    Returns ``[{"id", "S", "n_params"}]`` sorted by ``S`` descending.
    """
    batches = _as_batches(calibration, batch_size)
    units = units or list_units(model, _n_layers(model),
                                granularity=granularity)
    cfg = ScanConfig(bits_ref=bits, n_draws=n_draws, seed=seed,
                     batch_size=batch_size)
    res = run_lolo_scan(model, units, batches, cfg, bits=bits,
                        n_draws=n_draws, salt="sensitivity")
    S = res.seq_kl.mean(axis=1)
    rows = [{"id": u.id, "S": float(S[i]), "n_params": u.n_params}
            for i, u in enumerate(units)]
    return sorted(rows, key=lambda r: -r["S"])


def sensitivity_report(model, calibration, bits: int = 4, n_draws: int = 2,
                       null_bits: int = 8, n_null_draws: int = 1,
                       bootstrap_B: int = 2000, top_k: int = 8,
                       granularity: str = "block", seed: int = 20261014,
                       batch_size: int = 2, units=None) -> dict:
    """Full Stage-1 analysis: scan + null floor + BCa CIs + gates.

    Returns ``{"units": [{id, S, ci_low, ci_high, ci_method, p_topk,
    gate_pass, tail}], "null_floor": {...}, "split_half": {...}}`` —
    the same shape the optimizer's significance gate consumes.
    Per-unit CIs are marginal, not family-wise (METHODOLOGY.md §6.3).
    """
    batches = _as_batches(calibration, batch_size)
    units = units or list_units(model, _n_layers(model),
                                granularity=granularity)
    cfg = ScanConfig(bits_ref=bits, n_draws=n_draws, seed=seed,
                     batch_size=batch_size)
    scan = run_lolo_scan(model, units, batches, cfg, bits=bits,
                         n_draws=n_draws, salt="sensitivity")
    null = run_lolo_scan(model, units, batches, cfg, bits=null_bits,
                         n_draws=n_null_draws, salt="sensitivity-null")
    cis, p_topk = _bs.bootstrap_unit_cis(scan.seq_kl, B=bootstrap_B,
                                         seed=seed, top_k=top_k)
    floor_cis, _ = _bs.bootstrap_unit_cis(null.seq_kl, B=bootstrap_B,
                                          seed=seed + 9, top_k=top_k)
    floor_hi = max(c["high"] for c in floor_cis)
    floor_max = max(0.0, floor_hi)
    split_half = _bs.split_half_spearman(scan.seq_kl, seed=seed + 5)
    rows = []
    for u in range(len(units)):
        rows.append({
            "id": units[u].id,
            "S": float(cis[u]["estimate"]),
            "ci_low": cis[u]["low"], "ci_high": cis[u]["high"],
            "ci_method": cis[u]["method"], "a_hat": cis[u]["a_hat"],
            "p_topk": float(p_topk[u]),
            "gate_pass": bool(cis[u]["low"] > floor_max),
            "tail": scan.tail[u],
        })
    return {
        "units": rows,
        "null_floor": {"max_ci_high": float(floor_hi), "bits": null_bits,
                       "method": "bca", "B": bootstrap_B},
        "split_half": split_half,
        "config": {"bits": bits, "n_draws": n_draws,
                   "bootstrap_B": bootstrap_B, "top_k": top_k,
                   "granularity": granularity, "seed": seed},
    }
