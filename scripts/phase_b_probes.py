"""Phase B — Stage 3/5: behavioral probe battery on 3 model variants (CPU).

Variants (all deterministic; the "mock" closed loop — see P1_REPORT.md for
the real-llama.cpp boundary):
  * fp16        — reference model as-is
  * mock_base   — ALL Linear weights + embed exactly INT-quantized at b=4
                  (mock of the uniform-Q4_K_M baseline)
  * mock_recipe — per-unit exact INT-quant at the optimizer's bit allocation
                  (mock of recipe.sh output)

Writes hidden_files/phaseB_probes_<variant>.json with per-item
{id, category, generation, score, seconds}. Checkpointed per item so a
restart resumes. Paired design: identical prompts/order across variants.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import exactquant as eq
from bitblame import probes
from bitblame.config import ProbeConfig
from bitblame.modeling import _resolve, list_units, load_hf_model

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
MODEL_DIR = f"{ROOT}/data/models/SmolLM2-135M"
DATA = f"{ROOT}/data"


def apply_variant(model, units, variant: str, alloc: dict | None):
    """Return {weight_name: original_tensor} stash; model is modified in place."""
    stash = {}
    if variant == "fp16":
        return stash
    targets: list[tuple[str, int]] = []  # (weight_name, bits)
    if variant == "mock_base":
        for u in units:
            for wn in u.weight_names:
                targets.append((wn, 4))
    elif variant == "mock_recipe":
        assert alloc is not None
        for u in units:
            b = alloc[u.id]
            for wn in u.weight_names:
                targets.append((wn, b))
    else:
        raise ValueError(variant)
    seen = set()
    for wn, b in targets:
        if wn in seen:
            continue
        seen.add(wn)
        w = _resolve(model, wn)
        stash[wn] = w.detach().clone()
        w.copy_(eq.exact_quantize_weight(w, b))
    return stash


def restore_variant(model, stash: dict):
    for wn, orig in stash.items():
        _resolve(model, wn).copy_(orig)


def run_variant(model, tokenizer, items, variant, pcfg, alloc=None):
    out_path = f"{HID}/phaseB_probes_{variant}.json"
    done = {}
    if os.path.exists(out_path):
        done = {r["id"]: r for r in json.load(open(out_path))}
        print(f"resuming {variant}: {len(done)}/{len(items)} done", flush=True)
    stash = apply_variant(model, list_units(model, 30), variant, alloc)
    t0 = time.time()
    try:
        for i, item in enumerate(items):
            if item["id"] in done:
                continue
            t1 = time.time()
            gen = probes.generate_greedy(
                model, tokenizer, item["prompt"], item["max_new_tokens"])
            score = probes.score_item(item, gen,
                                      timeout_s=pcfg.humaneval_timeout_s)
            done[item["id"]] = {
                "id": item["id"], "category": item["category"],
                "generation": gen, "score": score,
                "seconds": round(time.time() - t1, 1),
            }
            if (i + 1) % 10 == 0 or i == len(items) - 1:
                with open(out_path, "w") as f:
                    json.dump(list(done.values()), f)
                el = time.time() - t0
                print(f"[{variant}] {len(done)}/{len(items)} "
                      f"elapsed {el/60:.1f}m", flush=True)
    finally:
        with open(out_path, "w") as f:
            json.dump(list(done.values()), f)
        restore_variant(model, stash)
    # verify restoration: one forward must match a fresh load? (spot check
    # via stash emptiness — full check is the fp16 re-run equality below)
    return list(done.values())


def main() -> None:
    torch.set_num_threads(2)
    pcfg = ProbeConfig()
    # CPU-time adaptation: GSM8K generations rarely need 256 tokens
    pcfg.max_new_tokens = 160  # (per-item cap still respected per category)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    items = probes.build_battery(DATA, tokenizer, pcfg)
    with open(f"{HID}/phaseB_battery.json", "w") as f:
        json.dump([{k: v for k, v in it.items() if k != "prompt"}
                   for it in items], f, indent=2)
    n_cat = {}
    for it in items:
        n_cat[it["category"]] = n_cat.get(it["category"], 0) + 1
    print(f"battery: {len(items)} items {n_cat}", flush=True)
    # per-category max tokens (spec §5.2: fixed per category)
    for it in items:
        if it["category"] == "gsm8k":
            it["max_new_tokens"] = 160

    model = load_hf_model(MODEL_DIR)
    units = list_units(model, 30)

    # sanity: fp16 variant must reproduce generation deterministically
    g1 = probes.generate_greedy(model, tokenizer, items[0]["prompt"], 8)
    g2 = probes.generate_greedy(model, tokenizer, items[0]["prompt"], 8)
    assert g1 == g2, "non-deterministic generation!"
    print("generation deterministic: OK", flush=True)

    alloc = None
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))
    alloc = {u["id"]: analysis["optimizer"]["alloc_bits"][u["id"]] for u in units}

    for variant in ("fp16", "mock_base", "mock_recipe"):
        t0 = time.time()
        res = run_variant(model, tokenizer, items, variant, pcfg, alloc)
        acc = sum(r["score"] for r in res) / len(res)
        print(f"[{variant}] done in {(time.time()-t0)/60:.1f}m  "
              f"accuracy={acc:.3f}", flush=True)

    # restoration check: fp16 generation after all variants == original
    g3 = probes.generate_greedy(model, tokenizer, items[0]["prompt"], 8)
    assert g3 == g1, "model weights NOT restored after variants!"
    print("weight restoration: OK", flush=True)
    print("PROBES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
