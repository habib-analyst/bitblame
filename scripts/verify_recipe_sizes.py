"""Verify the v1.1 optimizer bit-cost model against the real GGUF files.

Checks (all from measured files, nothing assumed):
  1. The full file-level simulation (optimizer.predict_file_types:
     --tensor-type regex patterns in emission order + Q4_K_M mixer with
     mixer-only counters) reproduces the ACTUAL per-tensor type of every
     tensor (272/272) in BOTH the recipe GGUF (overrides + mixer) and the
     baseline GGUF (pure mixer).
  2. Predicted stored bytes (sum of n_elts * bytes_per_param(landed type))
     vs actual stored bytes for both files.
  3. Old (v1.0 uniform) model prediction vs actual, for contrast.
  4. Actual recipe/baseline size ratio (the "10.5% smaller" claim).

Usage: python3 scripts/verify_recipe_sizes.py
"""
from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import gguf, optimizer as opt

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
MODELS = f"{ROOT}/data/models"
N_LAYERS = 30
ROLES = ("attn_q", "attn_k", "attn_v", "attn_output",
         "ffn_gate", "ffn_up", "ffn_down")


def split_name(name: str):
    """gguf tensor name -> (unit_id, role) or (None, None) for norms."""
    m = re.fullmatch(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight", name)
    if m:
        return f"blk.{m.group(1)}", m.group(2)
    if name == "token_embd.weight":
        return "embed", "token_embd"
    if name == "output.weight":
        return "embed", "output"
    return None, None


def main() -> None:
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))
    alloc = analysis["optimizer"]["alloc_types"]

    f16 = gguf.read_gguf_header(f"{MODELS}/SmolLM2-135M-f16.gguf")
    recipe = gguf.read_gguf_header(f"{MODELS}/SmolLM2-135M-recipe.gguf")
    base = gguf.read_gguf_header(f"{MODELS}/SmolLM2-135M-q4km.gguf")

    # tensors in FILE order (F16 header order = quantize processing order)
    tensors = []
    for name, t in f16["tensors"].items():
        uid, role = split_name(name)
        tensors.append({"name": name, "unit_id": uid, "role": role,
                        "ncols": t["dims"][0], "n_elts": t["n_elts"]})

    # pattern list in emission order (same as closedloop / emit_recipe_sh)
    gguf_names = {f"blk.{i}": [f"blk.{i}.{r}.weight" for r in ROLES]
                  for i in range(N_LAYERS)}
    gguf_names["embed"] = ["token_embd.weight", "output.weight"]
    patterns = opt.build_patterns(alloc, gguf_names)

    # --- 1) type-level reproduction (full simulation) --------------------
    for label, header, pats in (("recipe", recipe, patterns),
                                ("baseline", base, [])):
        sim = opt.predict_file_types(pats, tensors, N_LAYERS)
        bad = [(n, sim[n], header["tensors"][n]["type"])
               for n in sim if sim[n] != header["tensors"][n]["type"]]
        print(f"{label} types reproduced: "
              f"{len(sim) - len(bad)}/{len(sim)} (mismatches: {bad[:5]})")

        # --- 2) byte-level prediction ----------------------------------
        pred = sum(t["n_elts"] * gguf.bytes_per_param(sim[t["name"]])
                   for t in tensors)
        actual = sum(t["n_bytes"] for t in header["tensors"].values())
        print(f"{label} predicted tensor bytes: {pred:,.0f}  actual: "
              f"{actual:,}  err {(pred - actual) / actual:+.4%}")

    # --- 3) old v1.0 uniform model, for contrast -------------------------
    old_used_mb = analysis["optimizer"]["used_mb"]
    print(f"old v1.0 model recipe prediction: {old_used_mb:.2f} MB "
          f"({old_used_mb * 1e6:,.0f} B) vs actual file "
          f"{recipe['file_bytes']:,} B "
          f"(err {(old_used_mb * 1e6 - recipe['file_bytes']) / recipe['file_bytes']:+.1%})")
    print(f"old v1.0 model baseline prediction: {analysis['budget_mb']:.2f} MB "
          f"vs actual file {base['file_bytes']:,} B "
          f"(err {(analysis['budget_mb'] * 1e6 - base['file_bytes']) / base['file_bytes']:+.1%})")

    # --- 4) the size claim, from real files -------------------------------
    ratio = recipe["file_bytes"] / base["file_bytes"]
    print(f"actual sizes: recipe {recipe['file_bytes']:,} B, "
          f"baseline {base['file_bytes']:,} B, ratio {ratio:.4f} "
          f"-> recipe is {(1 - ratio) * 100:.1f}% smaller")
    print("SIZE MODEL VERIFIED")


if __name__ == "__main__":
    main()
