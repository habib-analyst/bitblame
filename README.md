# BitBlame

Dynamic quantization forensics for GGUF / llama.cpp models: measure *which layers* (units) lose accuracy when a model is quantized, and turn the measurement into a mixed-precision recipe — with confidence intervals on every number.

Positioning: firefly/mlx-optiq-class dynamic attribution for the llama.cpp/GGUF ecosystem. Not the first such tool — see [Related work](#related-work).

## What it measures

- **Per-unit sensitivity (blame score).** Perturb one unit's weights with quantization-calibrated mock noise, measure mean per-token KL divergence of the output distribution vs the FP16 reference, over calibration sequences. Unit = one transformer block by default; the embedding is its own unit.
- **Token-category blame.** Position-bucket, outlier-token, and domain KL breakdowns with bootstrap CIs.
- **Exact verification.** The top-8 units are re-measured with exact quantize→dequantize at 3/4/6 bits, so the cheap screen is checked against real quantizer math.
- **Recipe.** Significance-gated exact-DP multiple-choice knapsack over 6 real GGUF quant types, emitted as `--tensor-type` overrides for `llama-quantize` plus imatrix guidance.
- **Closed-loop validation.** Re-quantize with real `llama-quantize`, re-run a 250-item probe battery, and apply a pre-registered paired-bootstrap verdict.

Full spec: `METHODOLOGY.md` (v1.1). Full measured numbers: `P2_REPORT.md`.

## Status: Phase B complete (SmolLM2-135M, CPU). Phase C pending.

### Honest results — read this before the headline numbers

> **The quality-win claim is NOT established.** On the completed 250-item battery the BitBlame recipe scored 11.60% vs 11.60% for uniform Q4_K_M (mean paired delta 0.00 pp, 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338): verdict NO_SIGNIFICANT_DIFFERENCE. Non-inferiority at a 0.5 pp margin also FAILS — the confidence interval is too wide to certify it. The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

### Measured results (SmolLM2-135M, n = 250 probe items)

| Finding | Number |
|---|---|
| Screen vs exact ranking (top-8) | Spearman ρ = 0.9286 at 4-bit (p = 0.0009, n = 8); ρ = 0.9286 at 3-bit; ρ = 0.7857 at 6-bit |
| Split-half rank stability | ρ = 0.9754 (n_half = 8) |
| Significance gate | 31/31 units pass (null-floor max CI-high = 0.00307) |
| Concentration | Top-3 units (embed + blk.29 + blk.11) = 57.9% of total measured sensitivity; embedding alone 32.0% (S = 0.3674, 95% CI [0.3471, 0.3927]) |
| Closed loop | Harness exit 0; 272 tensors verified, 0 mismatches |
| Optimizer bit-cost model | Reproduces 272/272 tensor types and stored bytes exactly (0.0000% error) on both GGUFs |
| Recipe size | 94,373,920 B vs Q4_K_M baseline 105,453,856 B (ratio 0.8949) — 10.5% smaller |
| Recipe vs baseline accuracy | 11.60% = 11.60%; Δ 0.00 pp; 95% BCa CI [−4.8 pp, +4.0 pp]; p = 0.5338; 15 wins / 15 losses / 220 ties |
| FP16 reference accuracy | 12.80% |
| Verdict (pre-registered) | NO_SIGNIFICANT_DIFFERENCE; non-inferiority at 0.5 pp margin FAILS |
| imatrix agreement (corroboration, not blocking) | Spearman ρ = 0.2908 (p = 0.119, n = 30 blocks) — BELOW the ρ ≥ 0.5 bar; corroboration FAILS. Never claim imatrix corroboration. |

### What this does and does not mean

**Supported:** BitBlame localizes quantization sensitivity (top-3 units ≈ 58% of measured sensitivity, null-floor-gated, split-half ρ = 0.9754). The mock-noise screen predicts exact-quantization ranking (ρ = 0.9286 on the top-8). The closed loop works end-to-end through real `llama-quantize`. A BitBlame recipe produced a 10.5%-smaller GGUF that performed identically to uniform Q4_K_M on the full 250-item battery. The optimizer's bit-cost model is verified exact.

**Not supported — never claim these:** that BitBlame improves accuracy; that it reduces degradation or preserves quality at smaller size (non-inferiority FAILS — CI too wide); that it regresses retrieval (on needle the recipe matched the FP16 reference's count, 3/40 — the baseline scored above the reference by chance); that the ranking transfers to larger models (135M only, Phase C not run); that it is the first tool or has no prior art.

## How it works

```
STAGE 0  Cache FP16 logits            (1 forward pass per calibration seq)
STAGE 1  Mock-noise LOLO scan         (per-unit sensitivity + bootstrap CIs,
                                        null-floor run, split-half gate)
STAGE 2  Exact top-k verification     (faithful k-quant math, slope calibration)
STAGE 3  Behavioral probe battery     (FP16 vs baseline-quant, paired items)
STAGE 4  Bit-budget optimizer         (significance-gated exact-DP knapsack
                                        → llama.cpp overrides + imatrix)
STAGE 5  Closed-loop validation       (re-quantize, re-measure, pre-registered
                                        paired bootstrap test)
OUTPUT   blame_report.json · heatmap.png · recipe.sh
```

## Install

```bash
pip install --user .            # local install (CPU torch)
# or: pip install --user -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

```bash
export PATH="$HOME/.local/bin:$PATH"   # bitblame installs to ~/.local/bin/bitblame
bitblame scan --model <local-hf-checkpoint-dir|tiny> --calib hidden_files/calib.txt --out runs/scan.npz
bitblame probes --scan runs/scan.npz --model <same-model-as-scan> --out runs/probes.json
bitblame recipe --probes runs/probes.json --out runs/recipe.json
bitblame validate --recipe runs/recipe.json --model <same-model> --out runs/validation.json
bitblame report --runs runs/ --out runs/report.html
```

Notes on the interface (verified against `bitblame --help`):

- `--model` is a **local Hugging Face checkpoint directory**, or `tiny` for the built-in random transformer (no downloads). A **GGUF path is NOT accepted** — `scan`/`probes`/`validate` load torch checkpoints only. The GGUF / `llama-quantize` closed loop is `scripts/phase_b_closedloop.py`.
- `probes` needs `--model` too (Stage 2 exact verification re-loads the model), plus `--scan` (the scan NPZ) and `--calib` (defaults to the calibration file recorded in the scan).
- `recipe --budget` is in **megabytes (MB)**; omit it for the auto-budget (predicted size of the uniform-Q4_K_M baseline).
- `validate` needs `--model` too; `--data-dir` defaults to `data` (expects `mmlu/`, `gsm8k-*.parquet`, `humaneval-test.parquet` inside). `validate` is the **torch-level** closed loop (exact-quantized weights in torch; sizes reported as predicted weight bytes, **not** measured GGUF bytes). The probe-battery data files are **not** shipped in this repo (`data/` is gitignored) — point `--data-dir` at your local copy.

## Related work

| Prior art | What it does | What BitBlame adds |
|---|---|---|
| **firefly** (neelvad/firefly) | Per-layer sensitivity via real torchao quantization, relative-L1 divergence, recipe strategies with accuracy-bar gating, closed loop on the served checkpoint; HF/torchao/vLLM/SGLang. No GGUF, no bootstrap CIs. | GGUF/llama.cpp recipes, KL metric, CIs on every blame number, token-category blame |
| **mlx-optiq** | Per-layer KL sensitivity (exact quantize→dequantize round-trip) on a six-domain calibration mix; greedy knapsack; MLX only. More faithful sensitivity than BitBlame's noise screen; no CIs, no token-category blame, no documented closed loop. | Cheaper screen + exact Stage-2 verification, CIs, token-category blame, GGUF |
| **quantbench (stef41)** | Static header parsing only; never executes. | — |
| **quantbench (jamespa16)** | Executes (llama-server + HumanEval pass@1) but aggregate-only: one score per quant, no layer signal. | Layer attribution |
| **microant** (Code-byte404/microant) | Closest adjacent work. Whole-model GGUF/MLX capability ruler with bootstrap CIs: 250-case capability pack across quant tiers, per-tier deltas with bootstrap 95% CIs, and a `confirm` stage. Its "within-chain failure attribution" is a tier/chain damage breakdown, not per-layer blame. No sensitivity measurement, no bit allocation, no recipe emission; model runs need Apple Silicon. | Per-layer blame feeding allocation; measured recipes; token-category blame; CPU (non-Apple) path |
| **morshoto/qwen-meetup-tokyo exp_002** | Paired-bootstrap practical equivalence (Q4_K_M vs Q8_0 GGUF, ±10 pp margin at 95%). Whole-model comparison only; no attribution. | Methodological sibling for validation stats, not a threat |

BitBlame's differentiators: GGUF/llama.cpp recipes; paired-bootstrap CIs on per-layer attribution, significance-gating the allocation; token-category blame; exact-DP knapsack over 6 real GGUF quant types; closed-loop validation through real `llama-quantize`. (CIs on whole-model GGUF *measurement* exist — microant, morshoto. BitBlame's precise claim is narrower: paired-bootstrap CIs on per-layer attribution feeding the allocator.)

## Limitations

- **SmolLM2-135M only.** Nothing here claims the ranking (embed-dominated, blk.29/11/28) transfers to 1.7B+ models. Phase C (Qwen3-1.7B) is the transfer test; it has not run.
- **The battery is underpowered at this scale.** All three models are near the floor: GSM8K 1.7–3.3%, HumanEval 0% for everyone, copy 0% for everyone, overall FP16 accuracy 12.80%. A recipe win or loss of a few pp is undetectable here. The verdict is "no evidence of a difference," not "evidence of no difference."
- **Mock-noise screen is approximate.** The screen-vs-exact agreement (ρ = 0.9286) buys back confidence for the top-8, but the screen's ranking of the other 23 units is unverified.
- **CPU adaptations.** Calibration N=16/T=256 (spec §12 wants 32×512); split-half ρ = 0.9754 ≥ 0.9 rescues the ranking's reliability, but the smaller N/T is a deviation from spec.
- **Greedy decoding only; English calibration only.** No sampling-temperature, multilingual, or long-context generality is claimed.
- **Per-unit CIs are marginal, not family-wise.** The significance gate is a screening heuristic, not an FWER-controlled discovery procedure.
- **Needle:** on the full 40 needle items the recipe matched the FP16 reference's count (3/40); the baseline scored 7/40 by chance above the reference. Do not report "BitBlame regressed retrieval."
- **imatrix agreement: ρ = 0.2908 (Spearman, p = 0.119, n = 30 transformer blocks; embedding excluded — no imatrix entry for it), BELOW the spec bar ρ ≥ 0.5. Corroboration FAILS, not blocking.** BitBlame's blame ranking is therefore NOT corroborated by llama.cpp's imatrix importance on this model. Do not claim imatrix corroboration.
- **Recipe-emission wart:** the `output.weight` pattern (emitted for tied embeddings) regex-matches every `blk.*.attn_output.weight` — harmless here, dropped/anchored in P3.

## Layout

- `src/bitblame/` — the Python package (mock-noise screen, KL, bootstrap BCa, token-category blame, exact verification, DP optimizer, GGUF header verification)
- `scripts/` — Phase B harnesses (closed-loop, probe battery, size verification, final validation)
- `hidden_files/` — raw Phase A/B artifacts (scan, null floor, analysis, probe JSONs, logs)
- `METHODOLOGY.md` — methodology spec v1.1
- `P2_REPORT.md` — full Phase B report with every measured number
- `launch_drafts/` — DRAFT launch copy, do not post

## Links

- GitHub: https://github.com/habib-analyst/bitblame
- Hugging Face dataset (audit tables): https://huggingface.co/datasets/habib-analyst/bitblame-audit
- Hugging Face demo (Static Space, preloaded results — no live run): https://huggingface.co/spaces/habib-analyst/bitblame-demo
- Hugging Face profile: https://huggingface.co/habib-analyst

## Author

Habib Ur Rehman. License: MIT.
