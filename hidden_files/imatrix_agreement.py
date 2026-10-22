"""imatrix agreement diagnostic — METHODOLOGY.md §7.4.2 (spec v1.1).

Parses hidden_files/imatrix.dat (llama-imatrix output on the Phase-B
calibration mix) into per-tensor importance, aggregates to per-unit
importance, and computes the Spearman rank correlation with per-unit
blame scores S_u from hidden_files/phaseB_analysis.json.

Corroboration bar: pre-registered Spearman rho >= 0.5. The number is
reported either way; a below-bar value does not invalidate the
allocation but must be disclosed with the diverging tensors named
(§7.4.2). The embedding unit has no imatrix entry (llama-imatrix
collects matmul weights only), so the comparison is over the 30 block
units.

Writes hidden_files/imatrix_agreement.json. Does NOT modify any
measured artifact.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import imatrix as im

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"


def main() -> None:
    imx = im.read_imatrix(f"{HID}/imatrix.dat")
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))
    blame = {r["id"]: r["S"] for r in analysis["unit_rows"]}

    primary = im.agreement(imx, blame, variant="mean_in_sum2")
    robust = im.agreement(imx, blame, variant="sum_sq")
    # Aggregation-robustness: the below-bar result must not be an artifact
    # of how per-tensor importance is pooled into per-unit importance.
    # (mean == sum here: every block has exactly 7 tensors.)
    import re as _re
    from collections import defaultdict as _dd
    import numpy as _np
    from scipy import stats as _ss
    per_role = _dd(dict)
    for base, slot in imx["tensors"].items():
        m = _re.fullmatch(r"blk\.(\d+)\.(.*)\.weight", base)
        if m:
            per_role[f"blk.{m.group(1)}"][m.group(2)] = \
                im.tensor_importance(slot)
    units_sorted = sorted(per_role)
    ys = _np.array([blame[u] for u in units_sorted])
    agg_robustness = {}
    for name, agg in (("mean", _np.mean), ("max", _np.max),
                      ("median", _np.median)):
        xs = _np.array([agg(list(per_role[u].values())) for u in units_sorted])
        r = _ss.spearmanr(xs, ys)
        agg_robustness[name] = {"rho": float(r.statistic),
                                "p_value": float(r.pvalue)}
    per_role_rho = {}
    for role in sorted({r for u in units_sorted for r in per_role[u]}):
        xs = _np.array([per_role[u][role] for u in units_sorted])
        r = _ss.spearmanr(xs, ys)
        per_role_rho[role] = {"rho": float(r.statistic),
                              "p_value": float(r.pvalue)}
    result = {
        "imatrix": f"{HID}/imatrix.dat",
        "blame_source": f"{HID}/phaseB_analysis.json",
        "n_imatrix_tensors": len(imx["tensors"]),
        "metadata": {k: v for k, v in imx["metadata"].items()
                     if k != "imatrix.datasets"},
        "datasets": imx["metadata"].get("imatrix.datasets"),
        "primary": primary,
        "robustness_sum_sq": {
            "rho": robust["rho"], "p_value": robust["p_value"],
            "n_units": robust["n_units"], "meets_bar": robust["meets_bar"]},
        "robustness_aggregation": agg_robustness,
        "robustness_per_role": per_role_rho,
        "note": ("embed excluded: no imatrix entry for the embedding "
                 "(llama-imatrix collects matmul weights only). "
                 "Bar rho >= 0.5 is corroboration, not blocking (§7.4.2)."),
    }
    with open(f"{HID}/imatrix_agreement.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"imatrix tensors: {len(imx['tensors'])}")
    print(f"primary (mean_in_sum2): rho={primary['rho']:.4f} "
          f"p={primary['p_value']:.4g} n={primary['n_units']} "
          f"bar>=0.5: {'PASS' if primary['meets_bar'] else 'BELOW BAR'}")
    print(f"robustness (sum_sq): rho={robust['rho']:.4f} "
          f"p={robust['p_value']:.4g} n={robust['n_units']} "
          f"bar>=0.5: {'PASS' if robust['meets_bar'] else 'BELOW BAR'}")
    if not primary["meets_bar"]:
        worst = sorted(primary["per_unit"], key=lambda r: -r["rank_gap"])[:8]
        print("largest rank divergences (imatrix rank vs blame rank):")
        for r in worst:
            print(f"  {r['unit']:>8} imatrix_rank={r['rank_imatrix']:2d} "
                  f"blame_rank={r['rank_blame']:2d} gap={r['rank_gap']}")
    print("wrote hidden_files/imatrix_agreement.json")


if __name__ == "__main__":
    main()
