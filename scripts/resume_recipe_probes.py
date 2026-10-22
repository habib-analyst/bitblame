"""Resume-only runner for the missing recipe-arm probe items (P2 gap 1).

Rebuilds the 250-item battery identically to phase_b_gguf_probes.main(),
verifies the rebuilt item IDs match the stored phaseB_battery.json exactly
(aborts on any mismatch — paired design requires identical prompts), then
runs ONLY the recipe variant. run_variant() resumes from the existing
hidden_files/phaseB_gguf_recipe.json (200/250), so only the missing
50 items (needle/10..39 + copy/0..19) are measured.

Usage: python3 scripts/resume_recipe_probes.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")
sys.path.insert(0, "/home/hatch/workspace/bitblame/scripts")

import phase_b_gguf_probes as P
from bitblame import probes
from bitblame.config import ProbeConfig
from transformers import AutoTokenizer


def main() -> None:
    pcfg = ProbeConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        f"{P.MODELS}/SmolLM2-135M", local_files_only=True)
    items = probes.build_battery(f"{P.ROOT}/data", tokenizer, pcfg)
    for it in items:
        if it["category"] == "gsm8k":
            it["max_new_tokens"] = 160
    stored = json.load(open(f"{P.HID}/phaseB_battery.json"))
    assert [it["id"] for it in items] == [s["id"] for s in stored], (
        "rebuilt battery order differs from stored battery — refusing to "
        "run (paired design would break)")
    assert len(items) == 250
    print(f"battery verified identical: {len(items)} items", flush=True)
    res = P.run_variant(items, "recipe", pcfg, 18083)
    acc = sum(r["score"] for r in res) / len(res)
    print(f"[recipe] done  accuracy={acc:.3f}  n={len(res)}", flush=True)
    print("RECIPE PROBES RESUME COMPLETE", flush=True)


if __name__ == "__main__":
    main()
