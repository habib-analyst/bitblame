"""P2 final validation — recompute the verdict on the COMPLETED battery.

Reads the three GGUF probe files (250 paired items each), runs the
pre-registered paired test (§8.2: BCa B=10,000, same items) for the
primary endpoint + per-category breakdown, recomputes the screen-vs-exact
Spearman agreement and the split-half reliability from the raw scan
artifacts, and writes hidden_files/phaseB_validation.json.

All seeds are FIXED (no hash() — PYTHONHASHSEED randomization made the
old phase_b_validate.py per-category seeds non-reproducible).

Usage: python3 scripts/final_validate.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats as _ss

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import bootstrap as bs

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
CAT_SEEDS = {"mmlu": 20261014 + 11, "gsm8k": 20261014 + 23,
             "humaneval": 20261014 + 37, "needle": 20261014 + 51,
             "copy": 20261014 + 67}


def load(variant: str) -> dict:
    rows = json.load(open(f"{HID}/phaseB_gguf_{variant}.json"))
    return {r["id"]: r for r in rows}


def main() -> None:
    f16, base, recipe = load("f16"), load("q4km"), load("recipe")
    ids = sorted(set(f16) & set(base) & set(recipe))
    assert len(ids) == 250, f"expected 250 paired items, got {len(ids)}"
    print(f"paired items: {len(ids)}", flush=True)

    acc = {v: float(np.mean([d[i]["score"] for i in ids]))
           for v, d in (("f16", f16), ("q4km", base), ("recipe", recipe))}
    print("accuracy: " + ", ".join(f"{k}={v:.4f}" for k, v in acc.items()),
          flush=True)

    delta = np.array([recipe[i]["score"] - base[i]["score"] for i in ids])
    test = bs.paired_bootstrap_test(delta, B=10000, seed=20261014)
    closedloop = json.load(open(f"{HID}/closedloop.json"))
    sizes = closedloop["sizes"]
    size_ok = sizes["recipe"] <= 1.01 * sizes["q4km"]
    print(f"PRIMARY recipe-base: delta_bar={test['delta_bar']:.4f} "
          f"CI=[{test['ci_low']:.4f}, {test['ci_high']:.4f}] "
          f"p={test['p_value']:.4f} verdict={test['verdict']} "
          f"size_ok={size_ok} size_ratio={sizes['recipe']/sizes['q4km']:.4f}",
          flush=True)

    # secondary: recipe vs FP16, baseline vs FP16
    for label, a, b in (("recipe-f16", recipe, f16), ("base-f16", base, f16)):
        d = np.array([a[i]["score"] - b[i]["score"] for i in ids])
        t = bs.paired_bootstrap_test(d, B=10000, seed=20261014)
        print(f"{label}: delta={t['delta_bar']:.4f} "
              f"CI=[{t['ci_low']:.4f}, {t['ci_high']:.4f}] p={t['p_value']:.4f} "
              f"verdict={t['verdict']}", flush=True)

    # per-category deltas (secondary, reported not gating)
    cats: dict[str, list] = {}
    for i in ids:
        cats.setdefault(f16[i]["category"], []).append(i)
    per_cat = {}
    for c, members in sorted(cats.items()):
        d = np.array([recipe[i]["score"] - base[i]["score"] for i in members])
        t = bs.paired_bootstrap_test(d, B=10000, seed=CAT_SEEDS[c])
        per_cat[c] = {
            "n": len(members),
            "acc_f16": float(np.mean([f16[i]["score"] for i in members])),
            "acc_base": float(np.mean([base[i]["score"] for i in members])),
            "acc_recipe": float(np.mean([recipe[i]["score"] for i in members])),
            "delta_bar": t["delta_bar"], "ci_low": t["ci_low"],
            "ci_high": t["ci_high"], "p_value": t["p_value"],
            "verdict": t["verdict"],
        }
        print(f"  {c:10s} n={len(members):3d} f16={per_cat[c]['acc_f16']:.3f} "
              f"base={per_cat[c]['acc_base']:.3f} "
              f"recipe={per_cat[c]['acc_recipe']:.3f} "
              f"delta={t['delta_bar']:+.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}] "
              f"p={t['p_value']:.4f} {t['verdict']}", flush=True)

    # compression non-inferiority (secondary): CI lower > -0.005 at size <= 0.9x
    noninf = bool(test["ci_low"] > -0.005
                  and sizes["recipe"] <= 0.9 * sizes["q4km"])
    print(f"compression non-inferiority (margin -0.5pp): {noninf} "
          f"(CI low {test['ci_low']:.4f}, ratio "
          f"{sizes['recipe'] / sizes['q4km']:.4f})", flush=True)

    # screen-vs-exact agreement (spec v1.1 gate rho >= 0.7), recomputed
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))
    screen = {r["id"]: r["S"] for r in analysis["unit_rows"]}
    stage2 = analysis["stage2"]
    screen_exact = {}
    for bits in ("3", "4", "6"):
        xs = [screen[u] for u in stage2]
        ys = [stage2[u]["kl_by_bits"][bits] for u in stage2]
        rho = float(_ss.spearmanr(xs, ys).statistic)
        p = float(_ss.spearmanr(xs, ys).pvalue)
        screen_exact[bits] = {"rho": rho, "p_value": p, "n": len(xs)}
        print(f"screen-vs-exact rho at {bits}-bit: {rho:.4f} (p={p:.4f}, "
              f"n={len(xs)}) gate>=0.7: {rho >= 0.7}", flush=True)

    # split-half reliability (spec §2.3 gate rho >= 0.9), recomputed
    scan = np.load(f"{HID}/phaseB_scan.npz")
    sh = bs.split_half_spearman(scan["seq_kl"], seed=20261014 + 5)
    print(f"split-half rho: {sh['rho']:.4f} (n_half={sh['n_half']}) "
          f"gate>=0.9: {sh['gate_pass']}", flush=True)

    validation = {
        "n_items": len(ids),
        "accuracy": acc,
        "primary": test,
        "recipe_vs_f16": None,
        "size_recipe": sizes["recipe"],
        "size_base": sizes["q4km"],
        "size_ratio": sizes["recipe"] / sizes["q4km"],
        "size_ok_1p01": bool(size_ok),
        "compression_noninferiority": noninf,
        "per_category": per_cat,
        "criterion": "CI lower bound > 0 AND size(recipe) <= 1.01*size(base)",
        "success": bool(test["ci_low"] > 0 and size_ok),
        "screen_vs_exact": screen_exact,
        "split_half": sh,
        "seeds": {"primary": 20261014, "per_category": CAT_SEEDS},
    }
    with open(f"{HID}/phaseB_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    print("phaseB_validation.json written", flush=True)


if __name__ == "__main__":
    main()
