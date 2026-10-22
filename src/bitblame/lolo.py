"""LOLO mock-noise scan — Stage 0/1, §2.2 of METHODOLOGY.md (spec v1.0).

For each unit u (default: one transformer block), add fresh mock noise
(§2.1, b_ref=4) to ONLY unit u's weights, forward over the calibration
batches, and score with directional KL (§3) against the FP16 reference.
Reference logits are computed once per batch (Stage 0) and reused for
every unit — the reference never moves.

Memory discipline (this VM has ~0.6 GB free): per batch, the fp16
reference logits are held in RAM while all (unit, draw) perturbed passes
for that batch run; weights are stashed/restored around each perturbed
pass so the model is bit-identical between units.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import kl as klmod
from . import mockquant as mq
from .config import ScanConfig
from .modeling import Unit, _resolve, forward_logits, forward_with_hidden


@dataclass
class ScanResult:
    unit_ids: list[str]
    seq_kl: np.ndarray          # (U, N) per-sequence mean KL, draw-averaged
    token_kl: np.ndarray        # (U, N, T) per-token KL, draw-averaged
    draw_S: np.ndarray          # (U, R) per-draw blame scores (variance diagnostic)
    tail: list[dict]            # per-unit P50/P95/max over tokens
    outlier_mask: np.ndarray    # (N, T) bool: top-1% max-abs-activation tokens
    bits: int
    n_draws: int
    runtime_s: float
    batch_times_s: list[float] = field(default_factory=list)


def _stash_add_restore(model, unit: Unit, noises: dict[str, torch.Tensor]):
    """Context manager: add noise to unit weights, restore on exit."""
    stash: dict[str, torch.Tensor] = {}

    class _Ctx:
        def __enter__(self):
            with torch.no_grad():
                for name, nz in noises.items():
                    w = _resolve(model, name)
                    stash[name] = w.detach().clone()
                    w.add_(nz)
            return self

        def __exit__(self, *exc):
            with torch.no_grad():
                for name, orig in stash.items():
                    _resolve(model, name).copy_(orig)
            return False

    return _Ctx()


def run_lolo_scan(
    model: torch.nn.Module,
    units: list[Unit],
    batches: list[torch.Tensor],   # list of (B, T) input_ids
    cfg: ScanConfig,
    bits: int,
    n_draws: int,
    salt: str = "",
    progress=None,
    ckpt_path: str | None = None,
) -> ScanResult:
    """Run the LOLO scan. `batches` are the calibration sequences (§5.1).

    If `ckpt_path` is given, partial accumulators are saved after every
    batch and a matching checkpoint is resumed (deterministic: noise is
    seeded per (unit, draw, batch_idx), so skipping batches is exact).
    """
    torch.set_num_threads(2)
    U = len(units)
    N = sum(b.shape[0] for b in batches)
    T = batches[0].shape[1]
    nb = len(batches)
    seq_kl = np.zeros((U, N), dtype=np.float64)
    token_kl = np.zeros((U, N, T), dtype=np.float32)
    draw_S = np.zeros((U, n_draws), dtype=np.float64)
    outlier_mask = np.zeros((N, T), dtype=bool)
    batch_times: list[float] = []
    t0 = time.time()

    start_bi = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        if (list(ck["unit_ids"]) == [u.id for u in units]
                and int(ck["bits"]) == bits and int(ck["n_draws"]) == n_draws
                and ck["seq_kl"].shape == (U, N)):
            seq_kl = ck["seq_kl"].astype(np.float64)
            token_kl = ck["token_kl"].astype(np.float32)
            draw_S = ck["draw_S"].astype(np.float64)
            outlier_mask = ck["outlier_mask"].astype(bool)
            batch_times = list(ck["batch_times"])
            start_bi = int(ck["done_batches"])
            print(f"resuming scan from batch {start_bi}/{nb}", flush=True)

    def save_ckpt(done: int):
        if ckpt_path:
            np.savez_compressed(
                ckpt_path, unit_ids=np.array([u.id for u in units]),
                seq_kl=seq_kl, token_kl=token_kl, draw_S=draw_S,
                outlier_mask=outlier_mask, bits=np.array(bits),
                n_draws=np.array(n_draws),
                done_batches=np.array(done),
                batch_times=np.array(batch_times))

    seq_offset = sum(b.shape[0] for b in batches[:start_bi])
    for bi in range(start_bi, nb):
        batch = batches[bi]
        bt0 = time.time()
        B = batch.shape[0]
        # ---- Stage 0: reference forward (cached for this batch) ----
        ref_logits, hidden = forward_with_hidden(model, batch)
        # outlier tokens: top 1% by max-abs activation of final hidden state
        hmax = hidden.detach().float().abs().amax(dim=-1)  # (B, T)
        flat = hmax.reshape(-1)
        k = max(1, int(0.01 * flat.numel()))
        thr = torch.topk(flat, k, sorted=False).values.min().item()
        outlier_mask[seq_offset:seq_offset + B] = (hmax >= thr).cpu().numpy()

        for ui, unit in enumerate(units):
            weights = {n: _resolve(model, n) for n in unit.weight_names}
            for d in range(n_draws):
                noises = mq.apply_mock_noise(
                    weights, bits, cfg.group_size, unit.id, d, bi, salt)
                with _stash_add_restore(model, unit, noises):
                    pert_logits = forward_logits(model, batch)
                tkl = klmod.token_kl(ref_logits, pert_logits)  # (B, T) float64
                draw_S[ui, d] += tkl.mean() / len(batches)
                seq_kl[ui, seq_offset:seq_offset + B] += tkl.mean(axis=1) / n_draws
                token_kl[ui, seq_offset:seq_offset + B] += (
                    tkl / n_draws).astype(np.float32)
                del pert_logits, tkl
        del ref_logits, hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        seq_offset += B
        batch_times.append(time.time() - bt0)
        save_ckpt(bi + 1)
        if progress is not None:
            progress(bi + 1, nb, time.time() - t0)

    tail = [klmod.kl_stats(token_kl[u]) for u in range(U)]
    return ScanResult(
        unit_ids=[u.id for u in units],
        seq_kl=seq_kl, token_kl=token_kl, draw_S=draw_S, tail=tail,
        outlier_mask=outlier_mask, bits=bits, n_draws=n_draws,
        runtime_s=time.time() - t0, batch_times_s=batch_times,
    )
