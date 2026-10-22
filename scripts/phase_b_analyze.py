"""Phase B — analysis: bootstrap CIs, gates, categories, Stage 2, optimizer.

Reads hidden_files/phaseB_scan.npz + phaseB_null.npz; writes
hidden_files/phaseB_analysis.json + recipe.sh (emitted, not executed).

Steps (§6):
  1. Per-unit BCa 95% CIs (B=2000), joint resampling -> P(top-8).
  2. Null-floor gate: CI_low(S_u) > max(0, CI_high(floor)).
  3. Split-half reliability: Spearman rho >= 0.9 (else flag).
  4. Token-category means + BCa CIs (§4.2).
  5. Stage 2: exact quant at b={3,4,6} for top-8 (or all gated, whichever
     smaller), slope fit with shrinkage (§7.2). Needs the model + batches,
     so it re-runs the reference forward passes.
  6. Optimizer: exact DP knapsack (§7.3) -> recipe.sh (§7.4).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import bootstrap as bs
from bitblame import categories as cat
from bitblame import exactquant as eq
from bitblame import optimizer as opt
from bitblame.config import BootstrapConfig, OptimizerConfig
from bitblame.modeling import list_units, load_hf_model, forward_logits

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
MODEL_DIR = f"{ROOT}/data/models/SmolLM2-135M"


def main() -> None:
    t0 = time.time()
    bcfg = BootstrapConfig()
    ocfg = OptimizerConfig()

    scan = np.load(f"{HID}/phaseB_scan.npz")
    null = np.load(f"{HID}/phaseB_null.npz")
    unit_ids = [str(u) for u in scan["unit_ids"]]
    U = len(unit_ids)
    seq_kl, token_kl = scan["seq_kl"], scan["token_kl"]
    N, T = seq_kl.shape[1], token_kl.shape[2]
    print(f"loaded scan: U={U} N={N} T={T} "
          f"(scan {float(scan['runtime_s']):.0f}s, null {float(null['runtime_s']):.0f}s)",
          flush=True)

    # 1) per-unit CIs (joint resampling -> P(top-8))
    t1 = time.time()
    cis, p_topk = bs.bootstrap_unit_cis(seq_kl, B=bcfg.b_units,
                                       seed=bcfg.seed, top_k=8)
    floor_cis, _ = bs.bootstrap_unit_cis(null["seq_kl"], B=bcfg.b_units,
                                         seed=bcfg.seed + 9, top_k=8)
    print(f"bootstrap CIs: {time.time()-t1:.1f}s", flush=True)

    # 2) significance gate (§7.1)
    floor_hi = max(c["high"] for c in floor_cis)
    floor_max = max(0.0, floor_hi)
    S = np.array([c["estimate"] for c in cis])
    gated = [bool(c["low"] > floor_max) for c in cis]
    order = np.argsort(-S)
    print(f"null floor: max CI_high={floor_hi:.3e}; "
          f"gated {sum(gated)}/{U}", flush=True)

    # 3) split-half reliability (§2.3)
    sh = bs.split_half_spearman(seq_kl, seed=bcfg.seed + 5)
    print(f"split-half rho={sh['rho']:.3f} gate_pass={sh['gate_pass']}", flush=True)

    # draw-to-draw variance diagnostic (§2.2 step 4)
    draw_S = scan["draw_S"]  # (U, R)
    draw_cv = draw_S.std(axis=1) / np.clip(draw_S.mean(axis=1), 1e-300, None)

    # 4) token categories
    t1 = time.time()
    cat_means = cat.category_means(token_kl, scan["outlier_mask"])
    # category CIs: B=2000 on the cached per-token KL (CPU)
    cat_cis = cat.category_bootstrap_ci(token_kl, scan["outlier_mask"],
                                        B=bcfg.b_units, seed=bcfg.seed + 21)
    print(f"categories: {time.time()-t1:.1f}s", flush=True)

    unit_rows = []
    for u in range(U):
        unit_rows.append({
            "id": unit_ids[u],
            "S": float(S[u]),
            "ci_low": cis[u]["low"], "ci_high": cis[u]["high"],
            "ci_method": cis[u]["method"], "a_hat": cis[u]["a_hat"],
            "p_top8": float(p_topk[u]),
            "gate_pass": gated[u],
            "draw_cv": float(draw_cv[u]),
            "tail": {"mean": float(scan["tail_S"][u]),
                     "p50": float(scan["tail_p50"][u]),
                     "p95": float(scan["tail_p95"][u]),
                     "max": float(scan["tail_max"][u])},
            "categories": {
                c: {"mean": float(cat_means[c][u])} for c in cat_means},
        })
    category_table = {}
    for ui, u in enumerate(unit_ids):
        category_table[u] = {}
        for cname, ci in cat_cis[ui].items():
            category_table[u][cname] = {
                "mean": ci["estimate"], "ci_low": ci["low"],
                "ci_high": ci["high"], "method": ci["method"]}

    # 5) Stage 2 — exact verification for top-8 (or all gated, smaller)
    k = min(8, sum(gated)) if sum(gated) else 8
    topk_units = [unit_ids[i] for i in order[:k]]
    print(f"stage 2: top-{k} = {topk_units}", flush=True)
    torch.set_num_threads(2)
    model = load_hf_model(MODEL_DIR)
    units = list_units(model, n_layers=30)
    umap = {u.id: u for u in units}
    calib = json.load(open(f"{HID}/phaseB_calib.json"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    # rebuild identical calibration batches (deterministic from calib.json)
    import pandas as pd
    df = pd.read_parquet(f"{ROOT}/data/fineweb-edu-docs.parquet")
    text_col = "text" if "text" in df.columns else df.columns[0]
    seq_len = calib["seq_len"]
    seqs = []
    for m in calib["sequences"]:
        ids = tokenizer(str(df.iloc[m["doc_row"]][text_col]),
                        truncation=False)["input_ids"][:seq_len]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    bs_ = calib.get("batch_size", 2)
    batches = [torch.stack(seqs[i:i + bs_]) for i in range(0, len(seqs), bs_)]
    t1 = time.time()
    ref_batches = [forward_logits(model, b) for b in batches]
    print(f"ref forward for stage2: {time.time()-t1:.1f}s", flush=True)
    stage2 = {}
    for uid in topk_units:
        kb = eq.measure_exact_kl(model, umap[uid], ref_batches, batches,
                                 bits_list=(3, 4, 6))
        fit = eq.fit_slope(kb)
        stage2[uid] = fit
        print(f"  {uid}: " + ", ".join(f"b{b}={v:.3e}" for b, v in kb.items())
              + f" -> alpha={fit['alpha']:.3f} R2={fit['r2']:.3f}", flush=True)
    del model

    # 6) optimizer
    kunits = []
    for ui, u in enumerate(units):
        uid = u.id
        c_hat = float(S[ui])
        alpha = stage2[uid]["alpha"] if uid in stage2 else ocfg.alpha_theory
        kunits.append({"id": uid, "n_params": u.n_params, "c_hat": c_hat,
                       "alpha": alpha, "gated": gated[ui]})
    budget = sum(opt.unit_size_mb(k["n_params"], 4) for k in kunits)
    sol = opt.solve_knapsack(kunits, budget, ocfg)
    gguf = {u.id: u.gguf_names for u in units}
    recipe = opt.emit_recipe_sh(sol["alloc_types"], gguf)
    with open(f"{ROOT}/recipe.sh", "w") as f:
        f.write(recipe)
    print(f"optimizer: loss={sol['predicted_loss']:.3e} "
          f"used {sol['used_mb']:.2f}/{sol['budget_mb']:.2f} MB", flush=True)
    print("alloc: " + ", ".join(f"{k}={v}" for k, v in
                                sorted(sol["alloc_types"].items())), flush=True)

    analysis = {
        "unit_rows": unit_rows,
        "null_floor": {"max_ci_high": floor_hi, "bits": 8,
                       "method": "bca", "B": bcfg.b_units},
        "split_half": sh,
        "category_table": category_table,
        "stage2": stage2,
        "optimizer": sol,
        "budget_mb": budget,
        "runtime_s": time.time() - t0,
        "config": {"B": bcfg.b_units, "top_k": 8,
                   "n_seqs": N, "seq_len": T},
    }
    with open(f"{HID}/phaseB_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"ANALYSIS COMPLETE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
