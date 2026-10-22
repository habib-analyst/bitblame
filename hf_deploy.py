#!/usr/bin/env python3
"""Build + deploy BitBlame HF dataset (habib-analyst/bitblame-audit) and
STATIC Space (habib-analyst/bitblame-demo).

Pattern copied from Trainsect's deploy_static.py (verified working 2026-10-14):
- Auth: surrogate credential via dynamic_credentials helper, sent as
  ``Authorization: Bearer hsurr:...`` ONLY to huggingface.co.
- Transport: curl subprocess (Python HTTP clients hang on huggingface.co
  through this VM's egress proxy). Never prints, logs, or persists the
  real credential.
- Uploads: ONE commit per repo via the NDJSON commit endpoint
  (header line + one base64 file line per file).

Honesty: every published surface carries the full honesty block —
quality-win NOT established (NO_SIGNIFICANT_DIFFERENCE), imatrix
corroboration FAILS (rho 0.2908 < 0.5 bar), Phase C pending Oct 17 GPU
quota, SmolLM2-135M only, no transfer claims.
"""
import base64
import csv
import html
import io
import json
import os
import subprocess
import sys

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import dynamic_credential_entry  # noqa: E402

CRED = "custom.huggingface"
DATASET = "habib-analyst/bitblame-audit"
SPACE = "habib-analyst/bitblame-demo"
CURL = ["curl", "-s", "--max-time", "300"]
HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(HERE, "hf_staging")
DS_DIR = os.path.join(HF, "dataset")
SP_DIR = os.path.join(HF, "space")

HONESTY_MD = """## Honesty block — read this before the headline numbers

- **Quality-win claim NOT established.** On the completed 250-item battery the BitBlame recipe scored 11.60% vs 11.60% for uniform Q4_K_M (mean paired delta 0.00 pp, 95% BCa CI [-4.8 pp, +4.0 pp], p = 0.5338): verdict **NO_SIGNIFICANT_DIFFERENCE**. Non-inferiority at a 0.5 pp margin also FAILS — the confidence interval is too wide to certify it. The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win.
- **imatrix corroboration FAILS.** Spearman rho = 0.2908 (p = 0.119, n = 30 transformer blocks; embedding excluded — no imatrix entry for it), BELOW the spec bar rho >= 0.5. Never claim imatrix corroboration.
- **Phase C pending.** The transfer test on Qwen3-1.7B has not run (pending Kaggle GPU quota refresh, Oct 17). Nothing here claims the ranking transfers to larger models.
- **135M only.** All numbers are SmolLM2-135M (134.5M params, 30 blocks), CPU run, greedy decoding, English calibration only.
- **Battery underpowered at this scale.** FP16 reference accuracy is only 12.80%; GSM8K 1.7-3.3%, HumanEval 0% for everyone, copy 0% for everyone. The verdict is "no evidence of a difference," not "evidence of no difference."
- **Needle.** On the full 40 needle items the recipe matched the FP16 reference's count (3/40); the baseline scored 7/40 by chance above the reference. Do not report "BitBlame regressed retrieval."
"""

DATASET_CARD = """---
license: mit
pretty_name: BitBlame Phase B audit (SmolLM2-135M)
task_categories:
- text-generation
tags:
- quantization
- gguf
- interpretability
---

# BitBlame Phase B audit — SmolLM2-135M

Per-unit quantization-sensitivity attribution (blame scores with bootstrap CIs), the measured mixed-precision recipe, the 250-item probe-battery results, and the imatrix agreement check — from the BitBlame Phase B run (2026-10-14). Code + full report: [github.com/habib-analyst/bitblame](https://github.com/habib-analyst/bitblame). Demo: [huggingface.co/spaces/habib-analyst/bitblame-demo](https://huggingface.co/spaces/habib-analyst/bitblame-demo).

""" + HONESTY_MD + """
## Files

- `attribution.csv` — one row per unit (31): blame score S_u (mean per-token KL under 4-bit mock-noise perturbation), 95% BCa CI, share of total sensitivity, significance-gate pass, allocated bits/type from the exact-DP optimizer, imatrix importance + ranks (embedding: no imatrix entry).
- `probe_battery.csv` — per-category paired results (recipe vs Q4_K_M baseline): n, accuracies (FP16 / baseline / recipe), paired delta, 95% BCa CI, p-value, verdict.
- `validation.json` — full closed-loop validation object (primary test, per-category, sizes, screen-vs-exact, split-half).
- `imatrix_agreement.json` — full imatrix agreement analysis (primary + robustness variants).
- `recipe.sh` — the measured recipe as `llama-quantize --tensor-type` overrides. Note: the `output.weight=Q5_K` line is a no-op on this tied-embedding model (no `output.weight` tensor exists in the GGUF); unmentioned tensors fall back to the base type Q4_K_M.

## Method (one paragraph)

Per-unit sensitivity: perturb one unit's weights with quantization-calibrated mock noise, measure mean per-token KL divergence vs the FP16 reference over calibration sequences. Top-8 units re-measured with exact quantize→dequantize at 3/4/6 bits. Significance-gated exact-DP multiple-choice knapsack over 6 real GGUF quant types -> recipe. Closed-loop validation: re-quantize with real `llama-quantize`, re-run a 250-item probe battery, pre-registered paired-bootstrap verdict. Full spec: `METHODOLOGY.md` in the repo; full numbers: `P2_REPORT.md` in the repo.
"""


def _surrogate() -> str:
    return dynamic_credential_entry(CRED)["surrogate"]


def _curl(args):
    return subprocess.run(CURL + args, capture_output=True, text=True)


def _load():
    with open(os.path.join(HERE, "hidden_files", "phaseB_analysis.json")) as f:
        analysis = json.load(f)
    with open(os.path.join(HERE, "hidden_files", "phaseB_validation.json")) as f:
        validation = json.load(f)
    with open(os.path.join(HERE, "hidden_files", "imatrix_agreement.json")) as f:
        imatrix = json.load(f)
    with open(os.path.join(HERE, "recipe.sh")) as f:
        recipe_sh = f.read()
    return analysis, validation, imatrix, recipe_sh


def build_dataset(analysis, validation, imatrix, recipe_sh):
    os.makedirs(DS_DIR, exist_ok=True)
    im_by_unit = {r["unit"]: r for r in imatrix["primary"]["per_unit"]}
    total_s = sum(r["S"] for r in analysis["unit_rows"])
    alloc = analysis["optimizer"]["alloc_bits"]
    alloc_t = analysis["optimizer"]["alloc_types"]
    rows = sorted(analysis["unit_rows"], key=lambda r: -r["S"])
    with open(os.path.join(DS_DIR, "attribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "S_u", "ci_low", "ci_high", "ci_method", "share_of_total",
                    "gate_pass", "n_params", "alloc_bits", "alloc_type",
                    "imatrix_importance", "rank_imatrix", "rank_blame", "rank_gap"])
        for i, r in enumerate(rows, 1):
            im = im_by_unit.get(r["id"])
            w.writerow([r["id"], f"{r['S']:.10g}", f"{r['ci_low']:.10g}",
                        f"{r['ci_high']:.10g}", r["ci_method"],
                        f"{r['S'] / total_s:.6f}", r["gate_pass"],
                        r.get("n_params", ""), alloc.get(r["id"], ""),
                        alloc_t.get(r["id"], ""),
                        f"{im['imatrix_importance']:.10g}" if im else "",
                        im["rank_imatrix"] if im else "",
                        im["rank_blame"] if im else "",
                        im["rank_gap"] if im else ""])
    with open(os.path.join(DS_DIR, "probe_battery.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "acc_fp16", "acc_baseline_q4km", "acc_recipe",
                    "delta_pp_recipe_minus_baseline", "ci_low_pp", "ci_high_pp",
                    "p_value", "verdict"])
        for cat, d in sorted(validation["per_category"].items()):
            w.writerow([cat, d["n"], d["acc_f16"], d["acc_base"], d["acc_recipe"],
                        d["delta_bar"] * 100, d["ci_low"] * 100, d["ci_high"] * 100,
                        d["p_value"], d["verdict"]])
        p = validation["primary"]
        w.writerow(["OVERALL", validation["n_items"],
                    validation["accuracy"]["f16"], validation["accuracy"]["q4km"],
                    validation["accuracy"]["recipe"],
                    p["delta_bar"] * 100, p["ci_low"] * 100, p["ci_high"] * 100,
                    p["p_value"], p["verdict"]])
    im_clean = json.loads(json.dumps(imatrix).replace(
        "/home/hatch/workspace/bitblame/", ""))
    with open(os.path.join(DS_DIR, "imatrix_agreement.json"), "w") as f:
        json.dump(im_clean, f, indent=2)
    with open(os.path.join(DS_DIR, "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)
    with open(os.path.join(DS_DIR, "recipe.sh"), "w") as f:
        f.write(recipe_sh)
    with open(os.path.join(DS_DIR, "README.md"), "w") as f:
        f.write(DATASET_CARD)
    print(f"dataset staged: {sorted(os.listdir(DS_DIR))}")
    # sanity: spot-check the headline rows
    with open(os.path.join(DS_DIR, "attribution.csv")) as f:
        rd = list(csv.DictReader(f))
    assert len(rd) == 31, len(rd)
    assert rd[0]["unit"] == "embed" and rd[0]["S_u"].startswith("0.3674"), rd[0]
    assert rd[1]["unit"] == "blk.29", rd[1]
    print("attribution.csv checks OK (31 rows, embed top at S=0.3674)")


def build_heatmap(analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(SP_DIR, exist_ok=True)

    rows = sorted(analysis["unit_rows"], key=lambda r: r["S"])
    labels = [r["id"] for r in rows]
    s = np.array([r["S"] for r in rows])
    lo = np.array([r["S"] - r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] - r["S"] for r in rows])
    colors = ["#b91c1c" if u in ("embed", "blk.29", "blk.11") else "#2563eb"
              for u in labels]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(labels, s, xerr=[lo, hi], color=colors, ecolor="#333",
            capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xscale("log")
    ax.set_xlabel("Blame score S_u = mean per-token KL (log scale), 95% BCa CI")
    ax.set_title("BitBlame per-unit sensitivity — SmolLM2-135M (Phase B)\n"
                 "red = top-3 units (57.9% of total measured sensitivity)")
    fig.tight_layout()
    out = os.path.join(SP_DIR, "heatmap.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"heatmap: {out} ({os.path.getsize(out)} bytes)")
    assert os.path.getsize(out) > 10000
    return out

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px;color:#1a1a1a;line-height:1.6}
h1{font-size:1.9em;margin-bottom:0}
h2{margin-top:1.6em}
.tagline{font-size:1.1em;color:#444}
.banner{background:#fff8e1;border:1px solid #f0c929;border-radius:8px;padding:12px 16px;margin:16px 0}
.banner b{color:#7a5b00}
.warn{background:#fef2f2;border:1px solid #f0a0a0;border-radius:8px;padding:12px 16px;margin:16px 0}
.tabs{display:flex;gap:4px;border-bottom:2px solid #ddd;margin-top:20px;flex-wrap:wrap}
.tab-btn{background:none;border:none;padding:10px 18px;font-size:1em;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px}
.tab-btn.active{border-bottom-color:#2563eb;font-weight:700;color:#2563eb}
.tab{display:none}
.tab.active{display:block}
code{background:#f1f3f5;border-radius:4px;padding:2px 6px;font-size:.9em}
pre{background:#111827;color:#e5e7eb;border-radius:8px;padding:14px;overflow-x:auto;font-size:.85em;line-height:1.5}
pre.cmd{background:#f8fafc;color:#111827;border:1px solid #e2e8f0}
table{border-collapse:collapse;width:100%;margin:12px 0}
td,th{border:1px solid #e2e8f0;padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f8fafc}
img.heatmap{width:100%;border:1px solid #e2e8f0;border-radius:8px}
footer{margin-top:32px;color:#666;font-size:.9em;border-top:1px solid #eee;padding-top:12px}
"""

SPACE_README = """---
title: BitBlame Demo
emoji: 🔬
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
---

# BitBlame Demo

**Dynamic quantization forensics for GGUF / llama.cpp models.** Measure which layers lose accuracy when a model is quantized, and turn the measurement into a mixed-precision recipe — with confidence intervals on every number.

Preloaded Phase B results (Static Space, no live run): the per-unit blame heatmap, the measured recipe, and the audit results table for SmolLM2-135M. **Quality-win claim NOT established** — verdict NO_SIGNIFICANT_DIFFERENCE, imatrix corroboration fails (rho = 0.2908 < 0.5 bar). See the Honesty block on the page.

Code: [github.com/habib-analyst/bitblame](https://github.com/habib-analyst/bitblame) · Data: [huggingface.co/datasets/habib-analyst/bitblame-audit](https://huggingface.co/datasets/habib-analyst/bitblame-audit)
"""


def build_space(analysis, validation, imatrix, recipe_sh):
    os.makedirs(SP_DIR, exist_ok=True)
    rows = sorted(analysis["unit_rows"], key=lambda r: -r["S"])
    total_s = sum(r["S"] for r in analysis["unit_rows"])
    alloc_t = analysis["optimizer"]["alloc_types"]
    tr = "".join(
        f"<tr><td>{i}</td><td><b>{html.escape(r['id'])}</b></td>"
        f"<td>{r['S']:.4f}</td><td>[{r['ci_low']:.4f}, {r['ci_high']:.4f}]</td>"
        f"<td>{100 * r['S'] / total_s:.1f}%</td>"
        f"<td>{'yes' if r['gate_pass'] else 'no'}</td>"
        f"<td>{alloc_t.get(r['id'], '')}</td></tr>"
        for i, r in enumerate(rows, 1))
    cat_rows = ""
    for cat, d in sorted(validation["per_category"].items()):
        cat_rows += (f"<tr><td>{html.escape(cat)}</td><td>{d['n']}</td>"
                     f"<td>{100 * d['acc_f16']:.1f}%</td><td>{100 * d['acc_base']:.1f}%</td>"
                     f"<td>{100 * d['acc_recipe']:.1f}%</td>"
                     f"<td>{100 * d['delta_bar']:+.2f} pp</td>"
                     f"<td>[{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}] pp</td>"
                     f"<td>{html.escape(d['verdict'])}</td></tr>")
    p = validation["primary"]
    esc_recipe = html.escape(recipe_sh)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BitBlame — dynamic quantization forensics (SmolLM2-135M audit)</title>
<style>{CSS}</style></head>
<body>
<h1>🔬 BitBlame — dynamic quantization forensics</h1>
<p class="tagline"><b>Which layers lose accuracy when a model is quantized?</b>
BitBlame measures per-unit quantization sensitivity with confidence intervals, and turns the measurement into a
mixed-precision GGUF recipe. This page shows the preloaded Phase B audit on <b>SmolLM2-135M</b> — no live run.</p>
<div class="warn"><b>Honesty block — read this before the headline numbers.</b>
<ul>
<li><b>Quality-win claim NOT established.</b> Recipe 11.60% vs uniform Q4_K_M 11.60% (delta 0.00 pp, 95% BCa CI [-4.8 pp, +4.0 pp], p = 0.5338): verdict <b>NO_SIGNIFICANT_DIFFERENCE</b>. Non-inferiority at a 0.5 pp margin FAILS. Honest reading: same quality at 10.5% smaller on this one small model.</li>
<li><b>imatrix corroboration FAILS.</b> Spearman rho = 0.2908 (p = 0.119, n = 30 blocks), BELOW the spec bar rho &gt;= 0.5. Never claim imatrix corroboration.</li>
<li><b>Phase C pending</b> (Qwen3-1.7B transfer test, Oct 17 GPU quota). <b>135M only</b> — no transfer claims. Battery underpowered at this scale (FP16 reference 12.80%).</li>
<li><b>Needle:</b> recipe matched the FP16 reference count (3/40); baseline scored 7/40 by chance above the reference. Do not report "BitBlame regressed retrieval."</li>
</ul></div>
<div class="tabs">
<button class="tab-btn active" id="btn-tab-heat" onclick="showTab('tab-heat')">Blame heatmap</button>
<button class="tab-btn" id="btn-tab-recipe" onclick="showTab('tab-recipe')">Recipe</button>
<button class="tab-btn" id="btn-tab-audit" onclick="showTab('tab-audit')">Audit results</button>
<button class="tab-btn" id="btn-tab-method" onclick="showTab('tab-method')">Methodology</button>
</div>
<div class="tab active" id="tab-heat">
<h2>Per-unit blame — SmolLM2-135M</h2>
<img class="heatmap" src="heatmap.png" alt="Per-unit blame scores with 95% BCa confidence intervals">
<p>Top-3 units (embed + blk.29 + blk.11) = 57.9% of total measured sensitivity; embedding alone 32.0%.
Split-half rank stability rho = 0.9754; 31/31 units pass the significance gate (null-floor max CI-high = 0.00307).
Screen-vs-exact ranking agreement: Spearman rho = 0.9286 at 4-bit (top-8).</p>
<table><tr><th>#</th><th>unit</th><th>S_u</th><th>95% CI</th><th>share</th><th>gate</th><th>alloc</th></tr>{tr}</table>
</div>
<div class="tab" id="tab-recipe">
<h2>The measured recipe</h2>
<p>Exact-DP knapsack over 6 real GGUF quant types, emitted as <code>llama-quantize --tensor-type</code> overrides.
Resulting GGUF: 94,373,920 B vs Q4_K_M baseline 105,453,856 B (ratio 0.8949 — <b>10.5% smaller</b>).
Note: the <code>output.weight=Q5_K</code> line is a no-op on this tied-embedding model (no such tensor exists);
unmentioned tensors fall back to Q4_K_M. Full recipe file: <a href="https://huggingface.co/datasets/habib-analyst/bitblame-audit/blob/main/recipe.sh">recipe.sh in the dataset</a>.</p>
<pre class="cmd">{esc_recipe}</pre>
<p>Run it yourself: <code>pip install git+https://github.com/habib-analyst/bitblame.git</code> then
<code>bitblame scan / probes / recipe / validate / report</code> (see the repo README Quickstart).</p>
</div>
<div class="tab" id="tab-audit">
<h2>Audit results — pre-registered paired test (n = 250)</h2>
<table>
<tr><th>Comparison</th><td>Recipe vs Q4_K_M baseline</td></tr>
<tr><th>Accuracy</th><td>11.60% = 11.60%</td></tr>
<tr><th>Mean paired delta</th><td>0.00 pp (15 wins, 15 losses, 220 ties)</td></tr>
<tr><th>95% BCa CI</th><td>[-4.8 pp, +4.0 pp], p = 0.5338</td></tr>
<tr><th>Verdict</th><td><b>NO_SIGNIFICANT_DIFFERENCE</b>; non-inferiority at 0.5 pp margin FAILS</td></tr>
<tr><th>Size</th><td>94,373,920 B vs 105,453,856 B — 10.5% smaller, size gate (&#8804; 1.01x) passes</td></tr>
<tr><th>Closed loop</th><td>Harness exit 0; 272/272 tensor types verified, 0 mismatches</td></tr>
<tr><th>Optimizer bit-cost model</th><td>Reproduces 272/272 tensor types and stored bytes exactly (0.0000% error) on both GGUFs</td></tr>
<tr><th>imatrix agreement</th><td>Spearman rho = 0.2908 (p = 0.119, n = 30) — BELOW rho &gt;= 0.5 bar; corroboration FAILS</td></tr>
</table>
<h3>Per-category paired deltas (recipe − baseline)</h3>
<table><tr><th>category</th><th>n</th><th>FP16</th><th>Q4_K_M</th><th>recipe</th><th>delta</th><th>95% CI</th><th>verdict</th></tr>{cat_rows}</table>
<p>Needle note: the bootstrap mechanically flags SIGNIFICANT_REGRESSION, but recipe-vs-reference is 0 (both 3/40) —
the baseline scored above the FP16 reference by chance.</p>
</div>
<div class="tab" id="tab-method">
<h2>Methodology (summary)</h2>
<ol>
<li><b>Stage 0 — cache FP16 logits</b> (1 forward pass per calibration seq).</li>
<li><b>Stage 1 — mock-noise LOLO scan.</b> Perturb one unit's weights with quantization-calibrated noise; mean per-token KL vs FP16 reference. Bootstrap BCa CIs (B=2000), null-floor significance gate, split-half reliability gate.</li>
<li><b>Stage 2 — exact top-8 verification.</b> Re-measure top units with exact quantize&rarr;dequantize at 3/4/6 bits; power-law fits. Screen-vs-exact agreement gate rho &gt;= 0.7.</li>
<li><b>Stage 3 — behavioral probe battery.</b> 250 items (100 MMLU + 60 GSM8K + 30 HumanEval + 40 needle + 20 copy), FP16 vs baseline-quant, paired items.</li>
<li><b>Stage 4 — bit-budget optimizer.</b> Significance-gated exact-DP multiple-choice knapsack over 6 real GGUF quant types &rarr; <code>llama-quantize</code> overrides + imatrix guidance.</li>
<li><b>Stage 5 — closed-loop validation.</b> Re-quantize with real <code>llama-quantize</code>, re-measure, pre-registered paired-bootstrap verdict.</li>
</ol>
<h2>Limitations</h2>
<ul>
<li>SmolLM2-135M only; no transfer claims. Phase C (Qwen3-1.7B) pending.</li>
<li>Mock-noise screen is approximate; screen ranking of the 23 non-top-8 units is unverified.</li>
<li>Calibration N=16/T=256 (spec wants 32x512) — CPU adaptation; split-half rho = 0.9754 rescues ranking reliability.</li>
<li>Greedy decoding only; English calibration only. Per-unit CIs are marginal, not family-wise.</li>
<li>Related work: firefly, mlx-optiq, microant exist — BitBlame is not the first such tool. Its precise claim is narrower: paired-bootstrap CIs on per-layer attribution, significance-gating the allocation, for GGUF/llama.cpp.</li>
</ul>
<p>Full spec: <code>METHODOLOGY.md</code> in the repo. Full numbers: <code>P2_REPORT.md</code> in the repo.</p>
</div>
<footer>Source: <a href="https://github.com/habib-analyst/bitblame">github.com/habib-analyst/bitblame</a> ·
Data: <a href="https://huggingface.co/datasets/habib-analyst/bitblame-audit">bitblame-audit</a> · MIT license</footer>
<script>
function showTab(id){{document.querySelectorAll('.tab').forEach(function(t){{t.classList.remove('active');}});document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});document.getElementById(id).classList.add('active');document.getElementById('btn-'+id).classList.add('active');}}
</script>
</body></html>
"""
    index_path = os.path.join(SP_DIR, "index.html")
    with open(index_path, "w") as f:
        f.write(page)
    with open(os.path.join(SP_DIR, "README.md"), "w") as f:
        f.write(SPACE_README)
    print(f"space staged: index.html ({len(page)} bytes)")
    for needle in ["NO_SIGNIFICANT_DIFFERENCE", "0.2908", "94,373,920",
                   "Methodology", "output.weight", "no live run"]:
        assert needle in page, f"missing in index.html: {needle}"
    print("index.html content checks OK")


def repo_exists(kind: str, repo_id: str) -> bool:
    out = _curl(["-o", "/dev/null", "-w", "%{http_code}",
                 f"https://huggingface.co/api/{kind}/{repo_id}"])
    return out.stdout.strip() == "200"


def create_repo(kind: str, name: str, extra: dict) -> None:
    type_map = {"datasets": "dataset", "spaces": "space", "models": "model"}
    payload = json.dumps({"type": type_map.get(kind, kind), "name": name,
                          "private": False, **extra})
    out = _curl(["-X", "POST", "https://huggingface.co/api/repos/create",
                 "-H", f"Authorization: Bearer {_surrogate()}",
                 "-H", "Content-Type: application/json",
                 "-d", payload])
    print(f"create {kind}/{name} ->", out.stdout[:400])
    if out.returncode != 0 or '"error"' in out.stdout:
        raise RuntimeError(f"repo creation failed: {out.stdout[:400]}")


def upload_files(kind: str, repo_id: str, file_pairs, message: str) -> None:
    lines = [json.dumps({"key": "header",
                         "value": {"summary": message, "description": ""}})]
    for local_path, repo_path in file_pairs:
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        lines.append(json.dumps({"key": "file", "value": {
            "content": content_b64, "path": repo_path, "encoding": "base64"}}))
    out = _curl([
        "-X", "POST",
        f"https://huggingface.co/api/{kind}/{repo_id}/commit/main",
        "-H", f"Authorization: Bearer {_surrogate()}",
        "-H", "Content-Type: application/x-ndjson",
        "--data-binary", "\n".join(lines),
    ])
    body = out.stdout.strip()
    ok = out.returncode == 0 and "error" not in body
    print(f"commit {repo_id} ({len(file_pairs)} files) -> {'OK' if ok else 'CHECK: ' + body[:500]}")
    if not ok:
        raise RuntimeError(f"commit failed: {body[:500]}")


def main() -> None:
    analysis, validation, imatrix, recipe_sh = _load()
    build_dataset(analysis, validation, imatrix, recipe_sh)
    build_heatmap(analysis)
    build_space(analysis, validation, imatrix, recipe_sh)
    if repo_exists("datasets", DATASET):
        print(f"Dataset already exists: {DATASET}")
    else:
        create_repo("datasets", "bitblame-audit", {})
        print(f"Dataset created: https://huggingface.co/datasets/{DATASET}")
    ds_files = [(os.path.join(DS_DIR, fn), fn) for fn in
                ["attribution.csv", "probe_battery.csv", "validation.json",
                 "imatrix_agreement.json", "recipe.sh", "README.md"]]
    upload_files("datasets", DATASET, ds_files,
                 "BitBlame Phase B audit: attribution + probe battery + imatrix + recipe")
    if repo_exists("spaces", SPACE):
        print(f"Space already exists: {SPACE}")
    else:
        create_repo("spaces", "bitblame-demo", {"sdk": "static"})
        print(f"Space created: https://huggingface.co/spaces/{SPACE}")
    sp_files = [(os.path.join(SP_DIR, fn), fn) for fn in
                ["index.html", "README.md", "heatmap.png"]]
    upload_files("spaces", SPACE, sp_files,
                 "Static demo: preloaded Phase B heatmap + recipe + audit (honest numbers)")
    out = _curl([f"https://huggingface.co/datasets/{DATASET}"])
    print("verify dataset page ->", "OK" if "BitBlame Phase B audit" in out.stdout else "CHECK")
    out = _curl([f"https://huggingface.co/spaces/{SPACE}"])
    served = "NO_SIGNIFICANT_DIFFERENCE" in out.stdout or "BitBlame" in out.stdout
    print(f"verify space page -> {'OK' if served else 'CHECK (may need a moment to build)'}")
    print(f"Dataset: https://huggingface.co/datasets/{DATASET}")
    print(f"Space: https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
