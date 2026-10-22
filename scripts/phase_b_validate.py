"""Phase B — final validation + report assembly.

  1. Paired bootstrap test (§6.3, §8.2): per-item deltas
     (recipe - q4km baseline) over the 250 GGUF probe items; BCa 95% CI
     (B=10,000); verdict SIGNIFICANT_GAIN / NO_SIGNIFICANT_DIFFERENCE /
     SIGNIFICANT_REGRESSION. Pre-registered criterion: CI lower bound > 0
     AND size(recipe) <= 1.01 * size(base).
  2. Per-category delta breakdown (secondary endpoint).
  3. blame_report.json (§9) + heatmap.png.

All numbers carry CIs; effect sizes reported in percentage points.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import bootstrap as bs
from bitblame import report as rpt

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"


def load(variant: str) -> dict:
    rows = json.load(open(f"{HID}/phaseB_gguf_{variant}.json"))
    return {r["id"]: r for r in rows}


def main() -> None:
    f16 = load("f16")
    base = load("q4km")
    recipe = load("recipe")
    ids = sorted(set(f16) & set(base) & set(recipe))
    assert len(ids) == 250, f"expected 250 paired items, got {len(ids)}"
    print(f"paired items: {len(ids)}", flush=True)

    cats = {}
    for i in ids:
        cats.setdefault(f16[i]["category"], []).append(i)

    # primary endpoint: mean paired delta (recipe - base)
    delta = np.array([recipe[i]["score"] - base[i]["score"] for i in ids])
    acc = {v: float(np.mean([d[i]["score"] for i in ids]))
           for v, d in (("f16", f16), ("q4km", base), ("recipe", recipe))}
    print("accuracy: " + ", ".join(f"{k}={v:.3f}" for k, v in acc.items()),
          flush=True)

    test = bs.paired_bootstrap_test(delta, B=10000, seed=20261014)
    closedloop = json.load(open(f"{HID}/closedloop.json"))
    sizes = closedloop["sizes"]
    size_ok = sizes["recipe"] <= 1.01 * sizes["q4km"]
    print(f"delta_bar={test['delta_bar']:.4f} "
          f"CI=[{test['ci_low']:.4f}, {test['ci_high']:.4f}] "
          f"p={test['p_value']:.4f} verdict={test['verdict']} "
          f"size_ok={size_ok}", flush=True)

    # per-category deltas (secondary, reported not gating)
    per_cat = {}
    for c, members in sorted(cats.items()):
        d = np.array([recipe[i]["score"] - base[i]["score"] for i in members])
        t = bs.paired_bootstrap_test(d, B=10000,
                                     seed=20261014 + hash(c) % 997)
        per_cat[c] = {
            "n": len(members),
            "acc_f16": float(np.mean([f16[i]["score"] for i in members])),
            "acc_base": float(np.mean([base[i]["score"] for i in members])),
            "acc_recipe": float(np.mean([recipe[i]["score"] for i in members])),
            "delta_bar": t["delta_bar"],
            "ci_low": t["ci_low"], "ci_high": t["ci_high"],
            "p_value": t["p_value"], "verdict": t["verdict"],
        }
        print(f"  {c:10s} n={len(members):3d} base={per_cat[c]['acc_base']:.3f} "
              f"recipe={per_cat[c]['acc_recipe']:.3f} "
              f"delta={t['delta_bar']:+.3f} [{t['ci_low']:+.3f},{t['ci_high']:+.3f}]",
              flush=True)

    # interaction gap (§7.2): predicted sum of losses vs measured KL delta
    # (measured here via probe delta; KL-based gap in analysis notes)
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))

    validation = {
        "n_items": len(ids),
        "accuracy": acc,
        "primary": test,
        "size_recipe": sizes["recipe"],
        "size_base": sizes["q4km"],
        "size_ratio": sizes["recipe"] / sizes["q4km"],
        "size_ok_1p01": bool(size_ok),
        "per_category": per_cat,
        "criterion": ("CI lower bound > 0 AND size(recipe) <= 1.01*size(base)"),
        "success": bool(test["ci_low"] > 0 and size_ok),
    }
    with open(f"{HID}/phaseB_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    # blame_report.json (§9)
    cfg = {"model": "SmolLM2-135M", "phase": "B", "n_seqs": 16, "seq_len": 256,
           "bits_ref": 4, "n_draws": 2, "bootstrap_B": 2000,
           "probe_B": 10000, "top_k": 8}
    rep = rpt.build_report(
        spec_version="1.0",
        unit_rows=analysis["unit_rows"],
        null_floor=analysis["null_floor"],
        split_half=analysis["split_half"],
        category_table=analysis["category_table"],
        stage2=analysis["stage2"],
        optimizer=analysis["optimizer"],
        validation=validation,
        seeds={"scan": 20261014, "bootstrap": 20261014, "probes": 20261014},
        config=cfg,
        notes=[
            "Phase B on SmolLM2-135M (trained), CPU. Mock-noise screen "
            "(§2.1) + exact INT verification (Stage 2) + real llama.cpp "
            "closed loop (Stage 5, novel sub-claim v).",
            "Per-unit CIs are marginal, not family-wise (§6.3).",
            "llama.cpp rev " + closedloop.get("llama_cpp_rev", "?"),
        ],
    )
    rpt.save_report(rep, f"{ROOT}/blame_report.json")
    ok = rpt.render_heatmap(rep, f"{ROOT}/heatmap.png")
    print(f"blame_report.json written; heatmap png: {ok}", flush=True)
    print(rpt.ascii_bars(analysis["unit_rows"]), flush=True)


if __name__ == "__main__":
    main()
