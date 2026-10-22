"""Output artifacts — §9 of METHODOLOGY.md (spec v1.0).

blame_report.json (machine-readable) + heatmap.png (layers x roles with
CI whiskers; token-category KL bars). All artifacts versioned with the
spec version and config hash.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def build_report(
    spec_version: str,
    unit_rows: list[dict],     # per-unit: id, S, ci_low, ci_high, p_topk, ...
    null_floor: dict,
    split_half: dict,
    category_table: dict,      # unit_id -> {cat: {mean, ci_low, ci_high}}
    stage2: dict,              # unit_id -> fit_slope output
    optimizer: dict | None,
    validation: dict | None,
    seeds: dict,
    config: dict,
    notes: list[str],
) -> dict:
    return {
        "spec_version": spec_version,
        "config_hash": config_hash(config),
        "config": config,
        "units": unit_rows,
        "null_floor": null_floor,
        "split_half": split_half,
        "token_categories": category_table,
        "stage2_exact": stage2,
        "optimizer": optimizer,
        "validation": validation,
        "seeds": seeds,
        "notes": notes,
    }


def save_report(report: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def render_heatmap(report: dict, path: str) -> bool:
    """Layers (y) x unit (x) heatmap with CI whiskers + category bars.
    Returns False if matplotlib is unavailable (ASCII fallback by caller)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    units = report["units"]
    ids = [u["id"] for u in units]
    S = np.array([u["S"] for u in units])
    lo = np.array([u["ci_low"] for u in units])
    hi = np.array([u["ci_high"] for u in units])
    order = np.argsort(-S)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.35 * len(ids))))
    ax = axes[0]
    y = np.arange(len(ids))
    ax.barh(y, S[order], xerr=[S[order] - lo[order], hi[order] - S[order]],
            color="#4C78A8", ecolor="black", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([ids[i] for i in order], fontsize=7)
    ax.set_xlabel("blame score S_u (mean token KL, nats)")
    ax.set_title("Per-unit quantization sensitivity (95% BCa CI)")
    ax.invert_yaxis()

    ax2 = axes[1]
    cats = report.get("token_categories", {})
    # aggregate category means across units
    cat_names = ["pos_first25", "pos_mid50", "pos_last25", "outlier_top1pct"]
    vals = []
    for c in cat_names:
        v = [cats[uid][c]["mean"] for uid in cats if c in cats[uid]]
        vals.append(float(np.mean(v)) if v else float("nan"))
    ax2.bar(cat_names, vals, color="#F58518")
    ax2.set_ylabel("mean token KL (nats)")
    ax2.set_title("Token-category KL (mean over units)")
    ax2.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def ascii_bars(units: list[dict], width: int = 40) -> str:
    order = sorted(units, key=lambda u: -u["S"])
    mx = max(u["S"] for u in units) or 1.0
    lines = []
    for u in order:
        bar = "#" * max(1, int(width * u["S"] / mx)) if u["S"] > 0 else ""
        lines.append(
            f"  {u['id']:>10}  {u['S']:.3e} [{u['ci_low']:.2e}, {u['ci_high']:.2e}]  {bar}")
    return "\n".join(lines)
