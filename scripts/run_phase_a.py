"""Phase A — plumbing validation on a tiny random torch transformer.

Proves the code runs end-to-end (mock noise -> KL -> LOLO -> bootstrap ->
categories -> exact-quant -> optimizer -> recipe) before any real-model
compute is spent. Per METHODOLOGY.md §12: "Proves the code runs; proves
nothing about real models."
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import bootstrap as bs
from bitblame import categories as cat
from bitblame import exactquant as eq
from bitblame import kl as klmod
from bitblame import lolo
from bitblame import mockquant as mq
from bitblame import optimizer as opt
from bitblame import probes
from bitblame import report as rpt
from bitblame.config import BootstrapConfig, OptimizerConfig, ProbeConfig, ScanConfig
from bitblame.modeling import (
    TinyTransformer, _resolve, forward_logits, list_units,
)

torch.manual_seed(0)
torch.set_num_threads(2)

VOCAB, N_SEQS, T, BATCH = 256, 8, 32, 2


def main() -> None:
    t0 = time.time()
    print("== Phase A: tiny-model plumbing ==", flush=True)

    model = TinyTransformer(vocab=VOCAB, d=64, n_heads=4, n_layers=4,
                            dff=128, seed=7).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    units = list_units(model, n_layers=4)
    print(f"units: {[u.id for u in units]} (n={len(units)})", flush=True)

    rng = np.random.default_rng(7)
    batches = [torch.from_numpy(rng.integers(0, VOCAB, (BATCH, T))).long()
               for _ in range(N_SEQS // BATCH)]

    cfg = ScanConfig(n_seqs=N_SEQS, seq_len=T, batch_size=BATCH, n_draws=2)

    # --- sanity: identical forward -> KL exactly 0 ---
    r0 = forward_logits(model, batches[0])
    r1 = forward_logits(model, batches[0])
    k0 = klmod.token_kl(r0, r1).mean()
    assert k0 == 0.0, f"zero-noise KL != 0: {k0}"
    print(f"sanity: zero-noise KL = {k0:.1e} (exact)", flush=True)

    # --- Stage 1: LOLO scan (b=4), null floor (b=8) ---
    scan = lolo.run_lolo_scan(model, units, batches, cfg, bits=4,
                              n_draws=2, salt="phaseA")
    null = lolo.run_lolo_scan(model, units, batches, cfg, bits=8,
                              n_draws=1, salt="phaseA-null")
    S = scan.seq_kl.mean(axis=1)
    order = np.argsort(-S)
    print("ranking (b=4): " + ", ".join(
        f"{scan.unit_ids[i]}:{S[i]:.2e}" for i in order), flush=True)
    print(f"scan runtime: {scan.runtime_s:.1f}s", flush=True)

    # reproducibility: same seeds -> identical S
    scan2 = lolo.run_lolo_scan(model, units, batches, cfg, bits=4,
                               n_draws=2, salt="phaseA")
    assert np.array_equal(scan.seq_kl, scan2.seq_kl), "scan not reproducible!"
    print("reproducibility: identical seq_kl on re-run (bit-identical)", flush=True)

    # --- bootstrap CIs + gates ---
    bcfg = BootstrapConfig(b_units=200, seed=1)
    cis, p_topk = bs.bootstrap_unit_cis(scan.seq_kl, B=bcfg.b_units,
                                        seed=bcfg.seed, top_k=2)
    floor_cis, _ = bs.bootstrap_unit_cis(null.seq_kl, B=bcfg.b_units,
                                         seed=bcfg.seed + 9, top_k=2)
    floor_hi = max(c["high"] for c in floor_cis)
    gated = [cis[u]["low"] > max(0.0, floor_hi) for u in range(len(units))]
    print(f"null floor: max CI_high = {floor_hi:.2e}; gated units: "
          f"{[scan.unit_ids[u] for u in range(len(units)) if gated[u]]}", flush=True)
    sh = bs.split_half_spearman(scan.seq_kl, seed=3)
    print(f"split-half Spearman rho = {sh['rho']:.3f} (gate_pass={sh['gate_pass']})",
          flush=True)

    # --- token categories ---
    cat_means = cat.category_means(scan.token_kl, scan.outlier_mask)
    print("category means (unit blk.0): " + ", ".join(
        f"{k}={v[0]:.2e}" for k, v in cat_means.items()), flush=True)
    assert (scan.outlier_mask.sum() > 0), "no outlier tokens flagged"

    # --- Stage 2: exact quant on top-2 units ---
    ref_batches = [forward_logits(model, bch) for bch in batches]
    stage2 = {}
    for u in order[:2]:
        unit = units[u]
        kb = eq.measure_exact_kl(model, unit, ref_batches, batches,
                                 bits_list=(3, 4, 6))
        fit = eq.fit_slope(kb)
        stage2[unit.id] = fit
        print(f"stage2 {unit.id}: " +
              ", ".join(f"b{b}={v:.2e}" for b, v in kb.items()) +
              f" -> alpha={fit['alpha']:.3f} (R2={fit['r2']:.3f})", flush=True)
    # weights restored check
    r2 = forward_logits(model, batches[0])
    assert torch.equal(r0, r2), "model weights NOT restored after stage 2!"

    # --- optimizer + recipe ---
    ocfg = OptimizerConfig()
    kunits = [{"id": u.id, "n_params": u.n_params,
               "c_hat": float(S[i]),
               "alpha": stage2.get(u.id, {}).get("alpha", ocfg.alpha_theory),
               "gated": bool(gated[i])}
              for i, u in enumerate(units)]
    budget = sum(opt.unit_size_mb(k["n_params"], 4) for k in kunits)
    sol = opt.solve_knapsack(kunits, budget, ocfg)
    gguf = {u.id: u.gguf_names for u in units}
    recipe = opt.emit_recipe_sh(sol["alloc_types"], gguf)
    assert "--tensor-type" in recipe
    print(f"optimizer: predicted_loss={sol['predicted_loss']:.3e}, "
          f"used {sol['used_mb']:.3f}/{sol['budget_mb']:.3f} MB", flush=True)
    print("alloc: " + ", ".join(f"{k}={v}" for k, v in sol["alloc_types"].items()),
          flush=True)

    # --- probe machinery (synthetic-only; no downloads in Phase A) ---
    class _Tok:
        vocab_size = VOCAB

        def decode(self, ids):
            return " ".join(f"t{i}" for i in ids)

    pcfg = ProbeConfig()
    tok = _Tok()
    needles = probes.build_needle_items(4, seed=5)
    copies = probes.build_copy_items(2, seed=6, tokenizer=tok, vocab_size=VOCAB)
    assert probes.score_needle("x 12345 y", ["12345"]) == 1
    assert probes.score_copy("abc", "abc") == 1
    assert probes.score_mmlu("  C) ...", "C") == 1
    assert probes.score_gsm8k("so the answer is 42", "42") == 1
    he = probes.score_humaneval(
        "def add(a, b):\n    return ", "a + b",
        "def check(candidate):\n    assert candidate(1, 2) == 3", "add",
        timeout_s=5)
    assert he == 1, "humaneval sandbox failed on trivial item"
    he_fail = probes.score_humaneval(
        "def add(a, b):\n    return ", "a - b",
        "def check(candidate):\n    assert candidate(1, 2) == 3", "add",
        timeout_s=5)
    assert he_fail == 0
    print(f"probe machinery OK ({len(needles)} needle, {len(copies)} copy built; "
          f"sandbox pass/fail correct)", flush=True)

    # --- report assembly ---
    unit_rows = [{
        "id": scan.unit_ids[u], "S": float(S[u]),
        "ci_low": cis[u]["low"], "ci_high": cis[u]["high"],
        "ci_method": cis[u]["method"], "p_topk": float(p_topk[u]),
        "gate_pass": bool(gated[u]),
        "tail": scan.tail[u],
    } for u in range(len(units))]
    rep = rpt.build_report(
        spec_version="1.0", unit_rows=unit_rows,
        null_floor={"max_ci_high": floor_hi, "bits": 8},
        split_half=sh, category_table={}, stage2=stage2,
        optimizer=sol, validation=None,
        seeds={"scan": cfg.seed}, config={"phase": "A", "model": "tiny-random"},
        notes=["Phase A plumbing validation — random weights, not real behavior."])
    rpt.save_report(rep, "/home/hatch/workspace/bitblame/phaseA_report.json")
    ok_png = rpt.render_heatmap(rep, "/home/hatch/workspace/bitblame/phaseA_heatmap.png")
    print(f"report written (heatmap png: {ok_png})", flush=True)
    print(f"\nPHASE A GREEN in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
