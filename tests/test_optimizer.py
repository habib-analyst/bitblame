"""Optimizer tests — DP correctness + the llama.cpp remap model.

The remap test replays the exact verification from
``scripts/verify_recipe_sizes.py`` against the two real Phase-B GGUFs:
the simulation must reproduce 272/272 tensor types and stored bytes at
0.0000% error on both files. Nothing is mocked: headers are parsed
from the real files.
"""
import json
import os
import re

import pytest

from bitblame import gguf, optimizer as opt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HID = os.path.join(ROOT, "hidden_files")
MODELS = os.path.join(ROOT, "data", "models")
ROLES = ("attn_q", "attn_k", "attn_v", "attn_output",
         "ffn_gate", "ffn_up", "ffn_down")


def _phase_b_tensors():
    f16 = gguf.read_gguf_header(os.path.join(MODELS, "SmolLM2-135M-f16.gguf"))
    tensors = []
    for name, t in f16["tensors"].items():
        m = re.fullmatch(
            r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight",
            name)
        if m:
            uid, role = f"blk.{m.group(1)}", m.group(2)
        elif name == "token_embd.weight":
            uid, role = "embed", "token_embd"
        elif name == "output.weight":
            uid, role = "embed", "output"
        else:
            uid, role = None, None
        tensors.append({"name": name, "unit_id": uid, "role": role,
                        "ncols": t["dims"][0], "n_elts": t["n_elts"]})
    return f16, tensors


def test_fallback_type_demotions():
    # ncols % 256 != 0 -> K-quant demotion (tensor_type_fallback)
    assert opt.fallback_type("Q6_K", 576) == "Q8_0"
    assert opt.fallback_type("Q5_K", 576) == "Q5_1"
    assert opt.fallback_type("Q4_K", 576) == "Q5_0"
    assert opt.fallback_type("Q2_K", 576) == "Q4_0"
    assert opt.fallback_type("Q3_K", 576) == "Q4_0"
    # ncols % 256 == 0 -> request lands as-is
    assert opt.fallback_type("Q6_K", 1536) == "Q6_K"
    assert opt.fallback_type("Q2_K", 256) == "Q2_K"


def test_dp_picks_highest_sensitivity_upgrade():
    # 3 units, 8M params each. Budget allows exactly one 2->4-bit upgrade;
    # the DP must spend it on the highest-sensitivity unit (A).
    units = [
        {"id": "A", "n_params": 8_000_000, "c_hat": 1.0,
         "alpha": 1.3862943611198906, "gated": True},
        {"id": "B", "n_params": 8_000_000, "c_hat": 0.1,
         "alpha": 1.3862943611198906, "gated": True},
        {"id": "C", "n_params": 8_000_000, "c_hat": 0.01,
         "alpha": 1.3862943611198906, "gated": True},
    ]
    # 2-bit unit = 2.04 MB; 4-bit unit = 4.08 MB; budget = 3x2.04 + 2.04
    sol = opt.solve_knapsack(units, 8.16)
    assert sol["alloc_bits"] == {"A": 4, "B": 2, "C": 2}
    assert sol["used_mb"] == pytest.approx(8.16, abs=0.01)
    assert sol["solver"] == "exact_dp"


def test_dp_pins_ungated_units_at_default():
    units = [
        {"id": "A", "n_params": 8_000_000, "c_hat": 1.0,
         "alpha": 1.3862943611198906, "gated": True},
        {"id": "B", "n_params": 8_000_000, "c_hat": 0.1,
         "alpha": 1.3862943611198906, "gated": False},
    ]
    sol = opt.solve_knapsack(units, 100.0)
    assert sol["alloc_bits"]["B"] == 4  # pinned at default_bits


def test_dp_bin_rounding_repair_never_exceeds_budget():
    # Regression: the DP constrains ROUNDED bin counts, so the raw DP
    # solution can exceed the true budget (A@4 + B@6 bins to exactly
    # 2000 bins but totals 1.0003 MB > 1.0 MB budget). The solver must
    # repair by downgrading (here: A 4->3, the smallest loss increase
    # per MB saved) so used_mb <= budget_mb. Note the §7.2 clamp
    # (loss(2) == loss(3)) is why the raw DP favors B@6 (1 bin, loss
    # 0.146) over B@4.
    units = [
        {"id": "A", "n_params": 1_960_196, "c_hat": 1.0,
         "alpha": 1.3862943611198906, "gated": True},
        {"id": "B", "n_params": 784, "c_hat": 1.0,
         "alpha": 1.3862943611198906, "gated": True},
    ]
    sol = opt.solve_knapsack(units, 1.0)
    assert sol["alloc_bits"] == {"A": 3, "B": 6}
    assert sol["used_mb"] <= 1.0 + 1e-9


def test_remap_reproduces_both_ggufs_exactly():
    """272/272 tensor types + stored bytes at 0.0000% on both files."""
    analysis = json.load(open(os.path.join(HID, "phaseB_analysis.json")))
    alloc = analysis["optimizer"]["alloc_types"]
    f16, tensors = _phase_b_tensors()
    gguf_names = {f"blk.{i}": [f"blk.{i}.{r}.weight" for r in ROLES]
                  for i in range(30)}
    gguf_names["embed"] = ["token_embd.weight", "output.weight"]
    # legacy emission (tied_embeddings=False): matches the command that
    # built the Phase-B recipe file.
    patterns = opt.build_patterns(alloc, gguf_names)
    for label, fname, pats in (
            ("recipe", "SmolLM2-135M-recipe.gguf", patterns),
            ("baseline", "SmolLM2-135M-q4km.gguf", [])):
        header = gguf.read_gguf_header(os.path.join(MODELS, fname))
        sim = opt.predict_file_types(pats, tensors, 30)
        mismatches = [n for n in sim
                      if sim[n] != header["tensors"][n]["type"]]
        assert mismatches == [], (label, mismatches[:5])
        pred = sum(t["n_elts"] * gguf.bytes_per_param(sim[t["name"]])
                   for t in tensors)
        actual = sum(t["n_bytes"] for t in header["tensors"].values())
        assert (pred - actual) / actual == pytest.approx(0.0, abs=1e-9)


def test_tied_output_pattern_dropped_and_mixer_restored():
    """P3 wart fix: with tied_embeddings=True the phantom output.weight
    pattern is dropped, and mixer-block attn_output lands at the mixer
    tier (Q5_0) instead of fallback(embed alloc) = Q5_1."""
    analysis = json.load(open(os.path.join(HID, "phaseB_analysis.json")))
    alloc = analysis["optimizer"]["alloc_types"]
    _, tensors = _phase_b_tensors()
    gguf_names = {f"blk.{i}": [f"blk.{i}.{r}.weight" for r in ROLES]
                  for i in range(30)}
    gguf_names["embed"] = ["token_embd.weight", "output.weight"]
    legacy = opt.build_patterns(alloc, gguf_names)
    fixed = opt.build_patterns(alloc, gguf_names, tied_embeddings=True)
    assert any(p == "output.weight" for p, _ in legacy)
    assert not any(p == "output.weight" for p, _ in fixed)
    sim_legacy = opt.predict_file_types(legacy, tensors, 30)
    sim_fixed = opt.predict_file_types(fixed, tensors, 30)
    changed = [n for n in sim_legacy if sim_legacy[n] != sim_fixed[n]]
    # exactly the 8 mixer-block attn_output tensors (Q4_K units)
    assert len(changed) == 8
    assert all(n.endswith("attn_output.weight") for n in changed)
    assert all(sim_legacy[n] == "Q5_1" for n in changed)
    assert all(sim_fixed[n] == "Q5_0" for n in changed)
    sh = opt.emit_recipe_sh(alloc, gguf_names, tied_embeddings=True)
    assert "'output.weight=" not in sh
    assert "'token_embd.weight=" in sh
    sh_legacy = opt.emit_recipe_sh(alloc, gguf_names)
    assert "'output.weight=" in sh_legacy
