"""Command-line interface for BitBlame.

Entry point installed as ``bitblame`` (see pyproject.toml). Subcommands
map to the METHODOLOGY.md pipeline stages (all CPU-feasible on small
inputs; the SmolLM2-135M Phase-B artifacts in ``hidden_files/`` came
from the same modules via ``scripts/phase_b_*.py``):

- ``scan``     — Stage 0/1 LOLO mock-noise scan (+ null floor)
- ``probes``   — Stage-1 analysis (BCa CIs, gates, categories) + Stage-2
                 exact top-k verification probes
- ``recipe``   — Stage 4 exact-DP bit-budget optimizer -> recipe.json + recipe.sh
- ``validate`` — torch-level closed loop: behavioral battery on the
                 baseline vs recipe exact-quantized weights + paired test
- ``report``   — assemble blame_report.json + heatmap.png + report.html

Model argument (``--model``): a local Hugging Face checkpoint directory,
or ``tiny`` for the built-in random transformer (Phase-A plumbing; no
downloads, no tokenizer files needed).

Calibration (``--calib``): ``.txt`` (one document per line) or
``.parquet`` (a ``text`` column; FineWeb-Edu-style filtering: drop docs
with code fences or >5% non-ASCII). ``scan``/``probes`` rebuild the
identical calibration batches from this file.

Notes on the README Quickstart mapping:
- ``probes --scan`` takes the scan NPZ (Stage 1 output), not a probe
  file; ``recipe --probes`` takes the probes/analysis JSON.
- ``validate`` is the torch-level closed loop (exact-quantized weights
  in torch). The GGUF/``llama-quantize`` closed loop is
  ``scripts/phase_b_closedloop.py``; ``validate`` labels its sizes as
  predicted weight bytes, not measured GGUF bytes.
- ``recipe --budget`` is in megabytes (MB). Omit it for the default:
  the predicted size of the uniform-Q4_K_M baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

from . import bootstrap as bs
from . import categories as catmod
from . import exactquant as eq
from . import optimizer as opt
from . import probes as probemod
from . import report as reportmod
from .config import OptimizerConfig, ScanConfig
from .gguf import bytes_per_param
from .lolo import run_lolo_scan
from .modeling import TinyTransformer, list_units, load_hf_model

SPEC_VERSION = "1.1"


# --------------------------------------------------------------------------
# model + tokenizer + calibration helpers
# --------------------------------------------------------------------------

class CharTokenizer:
    """Byte-level tokenizer for the ``tiny`` model (vocab 256)."""

    vocab_size = 256

    def __call__(self, text, return_tensors=None, truncation=False):
        ids = [ord(c) % 256 for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i) % 256) for i in ids)


def load_model(model_arg: str):
    """Return (model, tokenizer, kind). kind is 'hf' or 'tiny'."""
    if model_arg == "tiny":
        return TinyTransformer(), CharTokenizer(), "tiny"
    if not os.path.isdir(model_arg):
        raise SystemExit(
            f"--model {model_arg!r}: not 'tiny' and not a local directory. "
            "BitBlame loads local HF checkpoints only (no downloads). "
            "A .gguf path cannot be scanned in torch — GGUF models are "
            "probed through llama.cpp (see scripts/phase_b_closedloop.py).")
    from transformers import AutoTokenizer
    model = load_hf_model(model_arg)
    tok = AutoTokenizer.from_pretrained(model_arg, local_files_only=True)
    return model, tok, "hf"


def model_n_layers(model, kind: str) -> int:
    if kind == "tiny":
        return int(model.config_n_layers)
    return int(model.config.num_hidden_layers)


def is_tied(model, kind: str) -> bool:
    """True when lm_head shares its tensor with embed_tokens."""
    if kind == "tiny":
        return False
    try:
        return model.lm_head.weight is model.model.embed_tokens.weight
    except AttributeError:
        return False


def load_calibration_texts(calib_path: str) -> list[str]:
    """Calibration documents from .txt (one doc/line) or .parquet."""
    if calib_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(calib_path)
        col = "text" if "text" in df.columns else df.columns[0]
        texts = []
        for _, row in df.iterrows():
            text = str(row[col])
            if "```" in text:
                continue
            if sum(ord(c) < 128 for c in text) / max(1, len(text)) < 0.95:
                continue
            texts.append(text)
        return texts
    with open(calib_path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def build_batches(tokenizer, texts: list[str], n_seqs: int, seq_len: int,
                  batch_size: int, seed: int = 20261014) -> list[torch.Tensor]:
    """Tokenize texts -> exactly n_seqs sequences of seq_len tokens."""
    seqs: list[torch.Tensor] = []
    for text in texts:
        if len(seqs) >= n_seqs:
            break
        ids = tokenizer(text, truncation=False)["input_ids"]
        if len(ids) < seq_len:
            continue
        seqs.append(torch.tensor(ids[:seq_len], dtype=torch.long))
    if len(seqs) < n_seqs:
        raise SystemExit(
            f"calibration: only {len(seqs)} usable sequences of length "
            f"{seq_len} (need {n_seqs}); use a longer calibration file or "
            "smaller --n-seqs/--seq-len.")
    return [torch.stack(seqs[i:i + batch_size])
            for i in range(0, len(seqs), batch_size)]


def role_shapes_for_units(model, units) -> dict[str, dict[str, tuple[int, int]]]:
    """{unit_id: {gguf_role: (n_params, ncols)}} for remap-aware sizing.

    ncols is the weight's input dim (torch shape[1]); it matches the
    GGUF header's dims[0] used by the fallback model.
    """
    from .modeling import _resolve

    out: dict[str, dict[str, tuple[int, int]]] = {}
    for u in units:
        shapes: dict[str, tuple[int, int]] = {}
        has_token_embd = "token_embd.weight" in u.gguf_names
        for wname, gname in zip(u.weight_names, u.gguf_names):
            w = _resolve(model, wname)
            m = re.fullmatch(r"blk\.(\d+)\.(.*)\.weight", gname)
            role = m.group(2) if m else (
                "token_embd" if gname == "token_embd.weight" else
                "output" if gname == "output.weight" else gname)
            if role == "output" and has_token_embd:
                continue  # tied lm_head: same tensor as token_embd
            ncols = int(w.shape[1]) if w.dim() >= 2 else int(w.shape[0])
            shapes[role] = (int(w.numel()), ncols)
        out[u.id] = shapes
    return out


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------

def cmd_scan(args) -> int:
    t0 = time.time()
    model, tokenizer, kind = load_model(args.model)
    n_layers = model_n_layers(model, kind)
    units = list_units(model, n_layers, granularity=args.granularity)
    if args.calib is None:
        if kind != "tiny":
            raise SystemExit("scan: --calib is required for HF models.")
        rng = np.random.default_rng(args.seed)
        seqs = [torch.tensor(rng.integers(0, 256, size=args.seq_len),
                             dtype=torch.long)
                for _ in range(args.n_seqs)]
        batches = [torch.stack(seqs[i:i + args.batch_size])
                   for i in range(0, len(seqs), args.batch_size)]
    else:
        texts = load_calibration_texts(args.calib)
        batches = build_batches(tokenizer, texts, args.n_seqs, args.seq_len,
                                args.batch_size, seed=args.seed)
    cfg = ScanConfig(bits_ref=args.bits, n_draws=args.n_draws,
                     n_seqs=args.n_seqs, seq_len=args.seq_len,
                     batch_size=args.batch_size, seed=args.seed,
                     granularity=args.granularity)
    print(f"scan: {len(units)} units, {args.n_seqs} seqs x {args.seq_len} "
          f"tokens, bits={args.bits}, draws={args.n_draws}", flush=True)
    scan = run_lolo_scan(model, units, batches, cfg, bits=args.bits,
                         n_draws=args.n_draws, salt="cli")
    null = None
    if not args.no_null:
        print(f"scan: null floor (bits={args.null_bits}, "
              f"draws={args.n_null_draws})", flush=True)
        null = run_lolo_scan(model, units, batches, cfg, bits=args.null_bits,
                             n_draws=args.n_null_draws, salt="cli-null")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "unit_ids": np.array([u.id for u in units]),
        "n_params": np.array([u.n_params for u in units]),
        "seq_kl": scan.seq_kl, "token_kl": scan.token_kl,
        "draw_S": scan.draw_S, "outlier_mask": scan.outlier_mask,
        "tail_S": np.array([t["mean"] for t in scan.tail]),
        "tail_p50": np.array([t["p50"] for t in scan.tail]),
        "tail_p95": np.array([t["p95"] for t in scan.tail]),
        "tail_max": np.array([t["max"] for t in scan.tail]),
        "bits": np.array(scan.bits), "n_draws": np.array(scan.n_draws),
        "runtime_s": np.array(scan.runtime_s),
        "batch_times": np.array(scan.batch_times_s),
        "meta": np.array([json.dumps({
            "model": args.model, "calib": args.calib,
            "n_seqs": args.n_seqs, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "seed": args.seed,
            "granularity": args.granularity, "tied": is_tied(model, kind)})]),
    }
    if null is not None:
        payload.update({
            "null_seq_kl": null.seq_kl, "null_token_kl": null.token_kl,
            "null_draw_S": null.draw_S, "null_bits": np.array(null.bits),
            "null_n_draws": np.array(null.n_draws),
        })
    np.savez_compressed(args.out, **payload)
    S = scan.seq_kl.mean(axis=1)
    order = np.argsort(-S)
    print(f"scan: wrote {args.out} in {time.time() - t0:.1f}s", flush=True)
    for i in order[:5]:
        print(f"  {scan.unit_ids[i]:>10}  S={S[i]:.4e}", flush=True)
    return 0


def load_scan(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta"][0])) if "meta" in d else {}
    return {"arrays": d, "meta": meta,
            "unit_ids": [str(u) for u in d["unit_ids"]]}


# --------------------------------------------------------------------------
# probes (analysis + Stage-2 exact verification)
# --------------------------------------------------------------------------

def cmd_probes(args) -> int:
    t0 = time.time()
    scan = load_scan(args.scan)
    A = scan["arrays"]
    unit_ids = scan["unit_ids"]
    U = len(unit_ids)
    seq_kl = np.asarray(A["seq_kl"], dtype=np.float64)
    token_kl = np.asarray(A["token_kl"], dtype=np.float64)
    if "null_seq_kl" not in A:
        raise SystemExit(f"{args.scan}: no null floor in scan file "
                         "(re-run scan without --no-null).")
    null_seq_kl = np.asarray(A["null_seq_kl"], dtype=np.float64)
    outlier_mask = np.asarray(A["outlier_mask"], dtype=bool)

    cis, p_topk = bs.bootstrap_unit_cis(seq_kl, B=args.bootstrap_b,
                                        seed=args.seed, top_k=args.top_k)
    floor_cis, _ = bs.bootstrap_unit_cis(null_seq_kl, B=args.bootstrap_b,
                                         seed=args.seed + 9,
                                         top_k=args.top_k)
    floor_hi = max(c["high"] for c in floor_cis)
    floor_max = max(0.0, floor_hi)
    split_half = bs.split_half_spearman(seq_kl, seed=args.seed + 5)
    print(f"probes: null floor max CI_high={floor_hi:.3e}; "
          f"split-half rho={split_half['rho']:.4f} "
          f"(gate>=0.9: {split_half['gate_pass']})", flush=True)
    draw_S = np.asarray(A["draw_S"], dtype=np.float64)
    draw_cv = (draw_S.std(axis=1)
               / np.clip(draw_S.mean(axis=1), 1e-300, None))
    cat_means = catmod.category_means(token_kl, outlier_mask)
    cat_cis = catmod.category_bootstrap_ci(token_kl, outlier_mask,
                                           B=args.bootstrap_b,
                                           seed=args.seed + 21)
    n_params = (np.asarray(A["n_params"]).tolist() if "n_params" in A
                else [0] * U)
    unit_rows = []
    for u in range(U):
        unit_rows.append({
            "id": unit_ids[u],
            "S": float(cis[u]["estimate"]),
            "ci_low": cis[u]["low"], "ci_high": cis[u]["high"],
            "ci_method": cis[u]["method"], "a_hat": cis[u]["a_hat"],
            "p_topk": float(p_topk[u]),
            "gate_pass": bool(cis[u]["low"] > floor_max),
            "draw_cv": float(draw_cv[u]),
            "n_params": int(n_params[u]),
            "tail": {"mean": float(A["tail_S"][u]),
                     "p50": float(A["tail_p50"][u]),
                     "p95": float(A["tail_p95"][u]),
                     "max": float(A["tail_max"][u])},
            "categories": {c: {"mean": float(cat_means[c][u])}
                           for c in cat_means},
        })
    category_table = {}
    for ui, uid in enumerate(unit_ids):
        category_table[uid] = {
            cname: {"mean": ci["estimate"], "ci_low": ci["low"],
                    "ci_high": ci["high"], "method": ci["method"]}
            for cname, ci in cat_cis[ui].items()}

    # Stage 2 — exact verification for top-k (or all gated, smaller)
    gated = [r["gate_pass"] for r in unit_rows]
    k = min(args.top_k, sum(gated)) if sum(gated) else args.top_k
    order = np.argsort([-r["S"] for r in unit_rows])
    topk_units = [unit_ids[i] for i in order[:k]]
    print(f"probes: stage 2 exact verification on top-{k}: {topk_units}",
          flush=True)
    model, tokenizer, kind = load_model(args.model)
    n_layers = model_n_layers(model, kind)
    units = list_units(model, n_layers,
                       granularity=scan["meta"].get("granularity", "block"))
    umap = {u.id: u for u in units}
    meta = scan["meta"]
    calib = meta.get("calib") or args.calib
    if calib is None:
        raise SystemExit("probes: Stage 2 needs --calib (or a scan whose "
                         "meta records one).")
    texts = load_calibration_texts(calib)
    batches = build_batches(tokenizer, texts, meta.get("n_seqs", 16),
                            meta.get("seq_len", 256),
                            meta.get("batch_size", 2), seed=args.seed)
    from .modeling import forward_logits
    torch.set_num_threads(2)
    ref_batches = [forward_logits(model, b) for b in batches]
    bits_list = tuple(args.bits)
    stage2: dict = {}
    for uid in topk_units:
        kb = eq.measure_exact_kl(model, umap[uid], ref_batches, batches,
                                 bits_list=bits_list)
        fit = eq.fit_slope(kb)
        stage2[uid] = fit
        print(f"  {uid}: " + ", ".join(f"b{b}={v:.3e}"
              for b, v in kb.items())
              + f" -> alpha={fit['alpha']:.3f} R2={fit['r2']:.3f}",
              flush=True)
    # screen-vs-exact agreement (spec §2.4 gate rho >= 0.7)
    screen_exact = {}
    if len(stage2) >= 3:
        from scipy import stats as _ss
        S_map = {r["id"]: r["S"] for r in unit_rows}
        for b in bits_list:
            xs = [S_map[u] for u in stage2]
            ys = [stage2[u]["kl_by_bits"][str(b) if str(b) in
                  stage2[u]["kl_by_bits"] else b] for u in stage2]
            res = _ss.spearmanr(xs, ys)
            screen_exact[str(b)] = {"rho": float(res.statistic),
                                    "p_value": float(res.pvalue),
                                    "n": len(xs),
                                    "gate_pass": bool(res.statistic >= 0.7)}
            print(f"probes: screen-vs-exact rho at {b}-bit: "
                  f"{res.statistic:.4f} (gate>=0.7: "
                  f"{screen_exact[str(b)]['gate_pass']})", flush=True)

    analysis = {
        "unit_rows": unit_rows,
        "null_floor": {"max_ci_high": float(floor_hi),
                       "bits": int(A["null_bits"]) if "null_bits" in A else 8,
                       "method": "bca", "B": args.bootstrap_b},
        "split_half": split_half,
        "category_table": category_table,
        "stage2": stage2,
        "screen_vs_exact": screen_exact,
        "role_shapes": role_shapes_for_units(model, units),
        "tied_embeddings": is_tied(model, kind),
        "config": {"bootstrap_B": args.bootstrap_b, "top_k": args.top_k,
                   "scan": args.scan, "model": args.model,
                   "exact_bits": list(bits_list)},
        "runtime_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"probes: wrote {args.out} in {time.time() - t0:.1f}s", flush=True)
    return 0

# --------------------------------------------------------------------------
# recipe
# --------------------------------------------------------------------------

def cmd_recipe(args) -> int:
    t0 = time.time()
    with open(args.probes) as f:
        analysis = json.load(f)
    ocfg = OptimizerConfig()
    n_layers = sum(1 for r in analysis["unit_rows"]
                   if re.fullmatch(r"blk\.(\d+)", r["id"]))
    role_shapes = analysis.get("role_shapes") or {}
    kunits = []
    for r in analysis["unit_rows"]:
        shapes = role_shapes.get(r["id"])
        unit = {"id": r["id"], "n_params": r.get("n_params", 0),
                "c_hat": r["S"],
                "alpha": analysis.get("stage2", {}).get(r["id"], {})
                        .get("alpha", ocfg.alpha_theory),
                "gated": bool(r["gate_pass"])}
        if shapes:
            unit["role_shapes"] = {k: tuple(v) for k, v in shapes.items()}
        kunits.append(unit)

    # budget default: predicted size of the uniform-Q4_K_M baseline
    # (remap model when shapes are known, else the v1.0 uniform model).
    if args.budget is not None:
        budget = float(args.budget)
    elif role_shapes:
        budget = sum(
            opt.role_bytes({k: tuple(v) for k, v in role_shapes[u["id"]].items()},
                           "Q4_K", opt.block_index(u["id"]), n_layers) / 1e6
            for u in kunits)
    else:
        budget = sum(opt.unit_size_mb(u["n_params"], 4) for u in kunits)
    # feasibility: every unit must fit at its lowest allowed option
    min_mb = 0.0
    for u in kunits:
        lo_bits = (min(ocfg.menu_bits) if u["gated"] else ocfg.default_bits)
        if u.get("role_shapes"):
            min_mb += opt.role_bytes(u["role_shapes"],
                                     ocfg.gguf_types[lo_bits],
                                     opt.block_index(u["id"]), n_layers) / 1e6
        else:
            min_mb += opt.unit_size_mb(u["n_params"], lo_bits)
    if budget < min_mb:
        raise SystemExit(
            f"recipe: budget {budget:.3f} MB is infeasible: minimum "
            f"feasible size is {min_mb:.3f} MB (all gated units at "
            f"{min(ocfg.menu_bits)} bits, ungated pinned at "
            f"{ocfg.default_bits} bits). Raise --budget.")
    sol = opt.solve_knapsack(kunits, budget, ocfg)
    tied = args.tied_embeddings
    if tied is None:
        tied = bool(analysis.get("tied_embeddings", False))
    # gguf names for emission
    gguf_names: dict[str, list[str]] = {}
    for r in analysis["unit_rows"]:
        uid = r["id"]
        if uid.startswith("blk."):
            i = uid.split(".")[1]
            gguf_names[uid] = [f"blk.{i}.{role}.weight" for role in
                               ("attn_q", "attn_k", "attn_v", "attn_output",
                                "ffn_gate", "ffn_up", "ffn_down")]
        elif uid == "embed":
            gguf_names[uid] = ["token_embd.weight", "output.weight"]
        elif uid == "lm_head":
            gguf_names[uid] = ["output.weight"]
        else:
            gguf_names[uid] = [uid]
    recipe_sh = opt.emit_recipe_sh(sol["alloc_types"], gguf_names,
                                   base_type=args.base_type,
                                   tied_embeddings=tied)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = {"alloc_bits": sol["alloc_bits"],
           "alloc_types": sol["alloc_types"],
           "predicted_loss": sol["predicted_loss"],
           "budget_mb": sol["budget_mb"], "used_mb": sol["used_mb"],
           "bin_mb": sol["bin_mb"], "n_bins": sol["n_bins"],
           "solver": sol["solver"],
           "base_type": args.base_type, "tied_embeddings": tied,
           "probes": args.probes,
           "units": [{"id": u["id"], "c_hat": u["c_hat"],
                      "alpha": u["alpha"], "gated": u["gated"]}
                     for u in kunits],
           "runtime_s": time.time() - t0}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if not args.no_sh:
        sh_path = args.sh or os.path.join(os.path.dirname(
            os.path.abspath(args.out)), "recipe.sh")
        with open(sh_path, "w") as f:
            f.write(recipe_sh)
        print(f"recipe: wrote {args.out} + {sh_path}", flush=True)
    else:
        print(f"recipe: wrote {args.out}", flush=True)
    print(f"recipe: predicted loss={sol['predicted_loss']:.3e} "
          f"used {sol['used_mb']:.2f}/{sol['budget_mb']:.2f} MB", flush=True)
    print("recipe: alloc: " + ", ".join(
        f"{k}={v}" for k, v in sorted(sol["alloc_types"].items())),
        flush=True)
    return 0


# --------------------------------------------------------------------------
# validate (torch-level closed loop)
# --------------------------------------------------------------------------

def _generate_torch(model, ids: torch.Tensor, max_new: int) -> list[int]:
    """Greedy generation loop for models without .generate (tiny)."""
    from .modeling import forward_logits
    cur = ids.clone()
    gen: list[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = forward_logits(model, cur)
            nxt = int(torch.argmax(logits[0, -1]).item())
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return gen


def _run_arm(model, tokenizer, kind: str, items: list[dict],
             max_new_cap: int | None) -> dict[str, dict]:
    """Run one arm (fp16 / baseline / recipe) over the battery items."""
    results: dict[str, dict] = {}
    for item in items:
        max_new = item["max_new_tokens"]
        if max_new_cap is not None:
            max_new = min(max_new, max_new_cap)
        if kind == "tiny":
            ids = torch.tensor([tokenizer(item["prompt"])["input_ids"]],
                               dtype=torch.long)
            gen_ids = _generate_torch(model, ids, max_new)
            generation = tokenizer.decode(gen_ids)
        else:
            generation = probemod.generate_greedy(model, tokenizer,
                                                  item["prompt"], max_new)
        score = probemod.score_item(item, generation, timeout_s=5)
        results[item["id"]] = {"score": score, "generation": generation,
                               "category": item["category"]}
    return results


def _apply_exact_quant(model, units, bits_by_unit: dict[str, int],
                       group_size: int = 256) -> dict:
    """Swap in exactly-quantized weights; return stash for restore."""
    from .modeling import _resolve
    stash: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for u in units:
            b = bits_by_unit.get(u.id)
            if b is None:
                continue
            for wname in u.weight_names:
                w = _resolve(model, wname)
                if wname not in stash:
                    stash[wname] = w.detach().clone()
                w.copy_(eq.exact_quantize_weight(stash[wname], b, group_size))
    return stash


def _restore(model, stash: dict):
    from .modeling import _resolve
    with torch.no_grad():
        for wname, orig in stash.items():
            _resolve(model, wname).copy_(orig)


def cmd_validate(args) -> int:
    t0 = time.time()
    with open(args.recipe) as f:
        recipe = json.load(f)
    alloc_bits: dict[str, int] = {k: int(v)
                                  for k, v in recipe["alloc_bits"].items()}
    model, tokenizer, kind = load_model(args.model)
    n_layers = model_n_layers(model, kind)
    units = list_units(model, n_layers)
    pcfg = probemod.ProbeConfig(seed=args.seed)
    data_dir = args.data_dir
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    items: list[dict] = []
    if "mmlu" in cats:
        items += probemod.build_mmlu_items(os.path.join(data_dir, "mmlu"),
                                           pcfg.n_mmlu, pcfg.seed)
    if "gsm8k" in cats:
        items += probemod.build_gsm8k_items(
            os.path.join(data_dir, "gsm8k-test.parquet"),
            os.path.join(data_dir, "gsm8k-train.parquet"),
            pcfg.n_gsm8k, pcfg.seed + 1)
    if "humaneval" in cats:
        items += probemod.build_humaneval_items(
            os.path.join(data_dir, "humaneval-test.parquet"),
            pcfg.n_humaneval, pcfg.seed + 2)
    if "needle" in cats:
        items += probemod.build_needle_items(pcfg.n_needle, pcfg.seed + 3)
    if "copy" in cats:
        items += probemod.build_copy_items(pcfg.n_copy, pcfg.seed + 4,
                                           tokenizer, tokenizer.vocab_size)
    if args.max_items is not None:
        # per-category cap, preserving the category mix
        kept: list[dict] = []
        per: dict[str, int] = {}
        for it in items:
            if per.get(it["category"], 0) < args.max_items:
                kept.append(it)
                per[it["category"]] = per.get(it["category"], 0) + 1
        items = kept
    if not items:
        raise SystemExit("validate: battery is empty (check --categories / "
                         "--max-items).")
    print(f"validate: {len(items)} battery items, "
          f"categories={sorted({i['category'] for i in items})}", flush=True)
    torch.set_num_threads(2)
    arms: dict[str, dict] = {}
    if not args.no_fp16:
        print("validate: arm fp16", flush=True)
        arms["fp16"] = _run_arm(model, tokenizer, kind, items,
                                args.max_new_tokens)
    print(f"validate: arm baseline (uniform {args.baseline_bits}-bit exact)",
          flush=True)
    stash = _apply_exact_quant(model, units,
                               {u.id: args.baseline_bits for u in units})
    try:
        arms["baseline"] = _run_arm(model, tokenizer, kind, items,
                                    args.max_new_tokens)
    finally:
        _restore(model, stash)
    print("validate: arm recipe (per-unit allocated bits)", flush=True)
    stash = _apply_exact_quant(model, units, alloc_bits)
    try:
        arms["recipe"] = _run_arm(model, tokenizer, kind, items,
                                  args.max_new_tokens)
    finally:
        _restore(model, stash)
    ids = sorted(set(arms["baseline"]) & set(arms["recipe"]))
    delta = np.array([arms["recipe"][i]["score"] - arms["baseline"][i]["score"]
                      for i in ids])
    test = bs.paired_bootstrap_test(delta, B=args.bootstrap_b, seed=args.seed)
    per_cat: dict[str, dict] = {}
    for c in sorted({arms["baseline"][i]["category"] for i in ids}):
        members = [i for i in ids if arms["baseline"][i]["category"] == c]
        d = np.array([arms["recipe"][i]["score"]
                      - arms["baseline"][i]["score"] for i in members])
        t = bs.paired_bootstrap_test(d, B=args.bootstrap_b,
                                     seed=args.seed + len(c))
        entry = {"n": len(members), "delta_bar": t["delta_bar"],
                 "ci_low": t["ci_low"], "ci_high": t["ci_high"],
                 "p_value": t["p_value"], "verdict": t["verdict"]}
        for arm in ("fp16", "baseline", "recipe"):
            if arm in arms:
                entry[f"acc_{arm}"] = float(np.mean(
                    [arms[arm][i]["score"] for i in members]))
        per_cat[c] = entry
    # predicted weight bytes (torch-level; NOT measured GGUF bytes)
    base_bits = args.baseline_bits
    pred = {}
    for label, bits_map in (("baseline", {u.id: base_bits for u in units}),
                            ("recipe", alloc_bits)):
        total = 0.0
        for u in units:
            b = bits_map.get(u.id, base_bits)
            total += u.n_params * bytes_per_param(
                OptimizerConfig().gguf_types[b])
        pred[label] = total
    validation = {
        "n_items": len(ids),
        "accuracy": {arm: float(np.mean([arms[arm][i]["score"] for i in ids]))
                     for arm in arms},
        "primary": test,
        "per_category": per_cat,
        "criterion": "CI lower bound > 0 (torch-level exact-quant loop; "
                     "sizes are predicted weight bytes, not GGUF bytes)",
        "success": bool(test["ci_low"] > 0),
        "predicted_weight_bytes": pred,
        "size_ratio_pred": (pred["recipe"] / pred["baseline"]
                            if pred["baseline"] else None),
        "recipe": args.recipe, "model": args.model,
        "categories": sorted({arms['baseline'][i]['category'] for i in ids}),
        "max_new_tokens_cap": args.max_new_tokens,
        "note": ("Torch-level closed loop: weights replaced by "
                 "exactly-quantized values (exactquant.py) per arm. The "
                 "GGUF/llama-quantize closed loop is "
                 "scripts/phase_b_closedloop.py."),
        "runtime_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"validate: delta_bar={test['delta_bar']:+.4f} "
          f"CI=[{test['ci_low']:+.4f},{test['ci_high']:+.4f}] "
          f"p={test['p_value']:.4f} verdict={test['verdict']}", flush=True)
    print(f"validate: wrote {args.out} in {time.time() - t0:.1f}s", flush=True)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(args) -> int:
    def _load(name: str, explicit, runs_file: str):
        if explicit:
            with open(explicit) as f:
                return json.load(f)
        p = os.path.join(args.runs, runs_file) if args.runs else None
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        if name == "probes":
            raise SystemExit("report: need probes.json — pass --probes or "
                             "--runs <dir containing probes.json>.")
        return None

    probes = _load("probes", args.probes, "probes.json")
    recipe = _load("recipe", args.recipe, "recipe.json")
    validation = _load("validation", args.validation, "validation.json")
    unit_rows = probes.get("unit_rows", [])
    rep = reportmod.build_report(
        spec_version=SPEC_VERSION,
        unit_rows=unit_rows,
        null_floor=probes.get("null_floor", {}),
        split_half=probes.get("split_half", {}),
        category_table=probes.get("category_table", {}),
        stage2=probes.get("stage2", {}),
        optimizer=recipe,
        validation=validation,
        seeds={"probes": 20261014},
        config=probes.get("config", {}),
        notes=[
            "BitBlame forensics report (METHODOLOGY.md spec v1.1).",
            "Per-unit CIs are marginal, not family-wise (§6.3).",
        ],
    )
    out_html = args.out
    stem, _ = os.path.splitext(out_html)
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    reportmod.save_report(rep, stem + ".json")
    png_ok = reportmod.render_heatmap(rep, stem + ".png")
    verdict = (validation or {}).get("primary", {}).get("verdict", "n/a")
    rows_html = "".join(
        f"<tr><td>{u['id']}</td><td>{u['S']:.4e}</td>"
        f"<td>[{u['ci_low']:.2e}, {u['ci_high']:.2e}]</td>"
        f"<td>{u.get('gate_pass')}</td></tr>" for u in unit_rows)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BitBlame report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}pre{{background:#f4f4f4;padding:1em;overflow-x:auto}}</style>
</head><body>
<h1>BitBlame forensics report</h1>
<p>Spec v{SPEC_VERSION} &middot; validation verdict: <b>{verdict}</b> &middot;
heatmap: {'<a href="' + os.path.basename(stem) + '.png">heatmap.png</a>' if png_ok else 'unavailable (matplotlib missing)'} &middot;
<a href="{os.path.basename(stem)}.json">blame_report.json</a></p>
<h2>Per-unit blame (S_u = mean token KL, 95% BCa CI)</h2>
<table><tr><th>unit</th><th>S_u</th><th>95% CI</th><th>gate</th></tr>{rows_html}</table>
<h2>Ranking (ASCII)</h2>
<pre>{reportmod.ascii_bars(unit_rows)}</pre>
</body></html>
"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"report: wrote {out_html} + {stem}.json"
          + (f" + {stem}.png" if png_ok else " (heatmap skipped)"),
          flush=True)
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bitblame",
        description="BitBlame — dynamic quantization forensics for the "
                    "llama.cpp/GGUF ecosystem (METHODOLOGY.md spec v1.1).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Stage 0/1: LOLO mock-noise scan + null floor",
                       description="Run the per-unit mock-noise sensitivity scan (Stage 1) and the null-floor run (§2.3). Writes one compressed NPZ with the scan, the null scan, and run metadata.")
    s.add_argument("--model", required=True, help="local HF checkpoint dir, or 'tiny'")
    s.add_argument("--calib", default=None, help=".txt (one doc/line) or .parquet calibration file (required for HF models)")
    s.add_argument("--out", required=True, help="output .npz path")
    s.add_argument("--n-seqs", type=int, default=16)
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=2)
    s.add_argument("--bits", type=int, default=4, help="b_ref for the mock-noise screen")
    s.add_argument("--n-draws", type=int, default=2, help="R independent noise draws per unit")
    s.add_argument("--null-bits", type=int, default=8)
    s.add_argument("--n-null-draws", type=int, default=1)
    s.add_argument("--no-null", action="store_true", help="skip the null-floor run (probes will refuse the scan)")
    s.add_argument("--granularity", default="block", choices=("block", "fine"))
    s.add_argument("--seed", type=int, default=20261014)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("probes", help="Stage-1 analysis + Stage-2 exact top-k probes",
                       description="Bootstrap BCa CIs, significance gate, split-half reliability, token-category blame (Stage 1 analysis) + exact quantize→dequantize verification of the top-k units (Stage 2). Writes the analysis JSON consumed by 'recipe'.")
    s.add_argument("--scan", required=True, help="scan .npz from 'bitblame scan'")
    s.add_argument("--model", required=True, help="same model the scan used (local HF dir or 'tiny')")
    s.add_argument("--calib", default=None, help="calibration file (defaults to the one recorded in the scan meta)")
    s.add_argument("--out", required=True, help="output analysis .json path")
    s.add_argument("--top-k", type=int, default=8)
    s.add_argument("--bits", type=int, nargs="+", default=[3, 4, 6], help="exact-verification bit-widths")
    s.add_argument("--bootstrap-b", type=int, default=2000)
    s.add_argument("--seed", type=int, default=20261014)
    s.set_defaults(func=cmd_probes)

    s = sub.add_parser("recipe", help="Stage 4: exact-DP bit-budget optimizer",
                       description="Solve the significance-gated multiple-choice knapsack (Stage 4) over the probes/analysis JSON. Writes recipe.json + recipe.sh (unless --no-sh).")
    s.add_argument("--probes", required=True, help="analysis .json from 'bitblame probes'")
    s.add_argument("--budget", type=float, default=None, help="budget in MB (default: predicted size of the uniform-Q4_K_M baseline)")
    s.add_argument("--out", required=True, help="output recipe .json path")
    s.add_argument("--sh", default=None, help="recipe.sh path (default: recipe.sh next to --out)")
    s.add_argument("--no-sh", action="store_true")
    s.add_argument("--base-type", default="Q4_K_M")
    s.add_argument("--tied-embeddings", action="store_true", default=None,
                   help="drop the phantom output.weight pattern (tied lm_head/embed). Default: use the flag recorded in the probes file.")
    s.set_defaults(func=cmd_recipe)

    s = sub.add_parser("validate", help="Torch-level closed-loop validation",
                       description="Behavioral probe battery on three arms (FP16, uniform-baseline exact-quant, recipe exact-quant) + pre-registered paired bootstrap test. Torch-level analogue of the §8 closed loop; the GGUF/llama-quantize loop is scripts/phase_b_closedloop.py.")
    s.add_argument("--recipe", required=True, help="recipe.json from 'bitblame recipe'")
    s.add_argument("--model", required=True, help="local HF checkpoint dir, or 'tiny'")
    s.add_argument("--data-dir", default="data", help="directory with mmlu/, gsm8k-*.parquet, humaneval-test.parquet")
    s.add_argument("--out", required=True, help="output validation .json path")
    s.add_argument("--categories", default="mmlu,gsm8k,humaneval,needle,copy",
                   help="comma list of battery categories to run")
    s.add_argument("--max-items", type=int, default=None, help="per-category item cap (default: full category)")
    s.add_argument("--max-new-tokens", type=int, default=None, help="cap on generated tokens per item")
    s.add_argument("--baseline-bits", type=int, default=4)
    s.add_argument("--no-fp16", action="store_true", help="skip the FP16 reference arm")
    s.add_argument("--bootstrap-b", type=int, default=10000)
    s.add_argument("--seed", type=int, default=20261014)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("report", help="Assemble blame_report.json + heatmap + HTML",
                       description="Assemble the §9 report artifacts from the probes/recipe/validation JSONs (explicit flags override --runs).")
    s.add_argument("--runs", default=None, help="directory containing probes.json / recipe.json / validation.json")
    s.add_argument("--probes", default=None)
    s.add_argument("--recipe", default=None)
    s.add_argument("--validation", default=None)
    s.add_argument("--out", required=True, help="output report .html path (JSON + PNG siblings are written alongside)")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
