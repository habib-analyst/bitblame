DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

DEV deep-dive — BitBlame: measuring which layers quantization breaks, with confidence intervals

# BitBlame: measuring which layers quantization breaks, with confidence intervals

Quantization usually works until it doesn't. A model drops points on one benchmark after conversion, and nobody knows which layer broke. The standard response is to try another quant, or another bit-width, and hope. BitBlame is an attempt to replace hoping with measuring: per-layer quantization blame for the llama.cpp/GGUF ecosystem, with a confidence interval on every number, and a closed loop that turns the measurement into a recipe and re-validates it.

This post covers the full Phase B run on SmolLM2-135M — including the null result, which is the most important part.

## The problem with heuristic allocation

llama.cpp's quantizer allocates bits by heuristics. Nothing in the toolchain measures which tensors actually take behavioral damage when quantized. Two existing tools do dynamic attribution for other ecosystems: firefly (torchao/vLLM, real kernels, binary allocation, no CIs, no GGUF) and mlx-optiq (MLX, KL sensitivity, greedy knapsack, no CIs, no documented closed loop). microant, the closest adjacent work, is a whole-model GGUF capability ruler with bootstrap CIs — but it does no per-layer blame, no bit allocation, and no recipe emission.

BitBlame's positioning is deliberately narrow: firefly/mlx-optiq-class dynamic attribution for the llama.cpp/GGUF ecosystem. Its differentiators are GGUF/llama.cpp recipes; paired-bootstrap CIs on per-layer attribution, significance-gating the allocation; token-category blame; an exact-DP knapsack over 6 real GGUF quant types; and closed-loop validation through real `llama-quantize`. (CIs on whole-model GGUF measurement exist — microant, morshoto. The precise claim is CIs on per-layer blame feeding the allocator.)

## Method: never pay for information twice

The pipeline has six stages:

- **Stage 0:** cache FP16 logits (1 forward pass per calibration sequence).
- **Stage 1 — mock-noise LOLO screen:** perturb one unit's weights at a time with uniform noise matched to quantization variance (per-group scale, k-quant super-block grouping), and measure mean per-token KL divergence of the output distribution vs the FP16 reference. R = 2 noise draws. BCa 95% CIs (B = 2000) over calibration sequences. A null-floor run (8-bit noise, near no-op) sets the significance gate, and a split-half reliability gate (ρ ≥ 0.9) checks the ranking.
- **Stage 2 — exact verification (blocking gate ρ ≥ 0.7):** the top-8 units are re-measured with faithful quantize→dequantize at 3/4/6 bits. Per-unit loss-vs-bit-width slopes are fit and shrunk toward the theoretical α = 2 ln 2 ≈ 1.386 (output KL is locally quadratic in the perturbation, so it decays as 2^(−2b)).
- **Stage 3 — probe battery:** 250 paired items (100 MMLU + 60 GSM8K + 30 HumanEval + 40 synthetic needle + 20 synthetic copy), identical prompts, greedy decoding, run on FP16, baseline-quant, and recipe-quant.
- **Stage 4 — optimizer:** significance-gated exact-DP multiple-choice knapsack over 6 GGUF quant types, emitted as `--tensor-type` overrides for `llama-quantize` plus imatrix guidance. Units that fail the significance gate are pinned at the default; the DP optimizes only over the gated set.
- **Stage 5 — closed-loop validation:** re-quantize with real `llama-quantize`, re-run the battery, pre-registered paired-bootstrap test (BCa, B = 10,000).

The mock-noise screen is approximate by design: real quantization error is deterministic and correlated with the weights, while the screen uses iid noise. Stage 2 exists because the screen is approximate. The question is whether the approximation preserves the ranking — and Phase B measured exactly that.

## What validated

**Screen vs exact agreement.** Screen-vs-exact Spearman ρ = 0.9286 on the top 8 at 4-bit (p = 0.0009, n = 8), ρ = 0.9286 at 3-bit, ρ = 0.7857 at 6-bit (KL values shrink toward the noise floor at 6-bit, so the lower ρ is expected). The blocking gate (ρ ≥ 0.7) passes. The single discordant adjacent pair is blk.27 vs blk.19 at 4-bit. The screen predicts the exact ranking on this model — the screen-fidelity risk is not the reason the closed loop failed to show a gain.

**Ranking reliability.** Split-half rank stability ρ = 0.9754 (n_half = 8), above the ρ ≥ 0.9 gate. All 31 units pass the significance gate (null-floor max CI-high = 0.00307). The top-7 units each have P(top-8) = 1.00 under bootstrap resampling.

**Concentration.** The top-3 units (embedding + blk.29 + blk.11) account for 57.9% of total measured sensitivity. The embedding alone is 32.0% (S = 0.3674, 95% CI [0.3471, 0.3927]), followed by blk.29 (S = 0.1604) and blk.11 (S = 0.1366). Among transformer blocks only, the top 3 (blk.29, blk.11, blk.28) account for 46.5% of block sensitivity. The concentration is real, not a CI artifact: the top-3 CIs sit far above the null floor.

**Optimizer fidelity.** The v1.0 bit-cost model assumed a requested tier lands uniformly and was off by −27.3% / −35.0% vs the real files. The v1.1 model reproduces llama.cpp's actual behavior — K-quant fallback demotion (Q2_K/Q3_K→Q4_0, Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0 on ncols % 256 ≠ 0), the Q4_K_M mixer with `use_more_bits`, mixer-only counters, and `--tensor-type` regex first-match semantics — and reproduces both GGUFs' 272/272 tensor types and stored bytes exactly (0.0000% error). The closed-loop harness, fixed to verify via direct GGUF header reads instead of the `llama-gguf` example tool's self-test, re-ran exit 0 with 272 tensors verified, 0 mismatches.

## The null result

The recipe came out 10.5% smaller than the uniform Q4_K_M baseline (94,373,920 B vs 105,453,856 B, ratio 0.8949). On the completed 250-item battery:

- FP16: 12.80%. Q4_K_M: 11.60%. Recipe: 11.60%.
- Mean paired delta: 0.00 pp (15 wins, 15 losses, 220 ties — the zero is exact, not a rounding artifact).
- 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338. Verdict: NO_SIGNIFICANT_DIFFERENCE.
- Non-inferiority at the 0.5 pp margin FAILS: the CI lower bound (−4.8 pp) is far below the −0.5 pp margin. The CI is simply too wide to certify non-inferiority.

[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] The quality-win claim is NOT established. The honest statement is: a 10.5% smaller model that performed identically to Q4_K_M on the full 250-item battery — not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

Per-category detail: MMLU +5.00 pp (CI [−5.0 pp, +14.0 pp]), GSM8K −1.67 pp (CI [−10.0 pp, 0.0 pp]), HumanEval 0.00 pp (all zero — floor), copy 0.00 pp (all zero — floor), needle −10.00 pp (CI [−27.5 pp, −5.0 pp]). The needle number needs its footnote: on the full 40 needle items, FP16 itself scores 3/40 (7.5%), and the recipe also scores 3/40 (7.5%) — the same count as the reference. The baseline's 7/40 (17.5%) is 4 items above the FP16 reference. The bootstrap mechanically flags a regression because it only sees recipe-vs-baseline, but recipe-vs-reference is 0. Do not report "BitBlame regressed retrieval."

## Limitations, stated plainly

- **SmolLM2-135M only.** Nothing here claims the ranking transfers to 1.7B+ models. Phase C is the transfer test; it has not run.
- **The battery is underpowered at this scale.** FP16 itself scores 12.80%; GSM8K 1.7–3.3%, HumanEval 0% for everyone, copy 0% for everyone. A recipe win or loss of a few pp is undetectable here. The verdict is "no evidence of a difference," not "evidence of no difference."
- **Mock-noise screen is approximate.** The top-8 ranking is verified exactly; the other 23 units are not.
- **CPU adaptations.** Calibration N=16/T=256 (spec wants 32×512); the split-half gate rescues the ranking's reliability, but the smaller N/T is a deviation from spec.
- **Greedy decoding only; English calibration only.** No sampling-temperature, multilingual, or long-context generality is claimed.
- **Per-unit CIs are marginal, not family-wise.** The significance gate is a screening heuristic, not an FWER-controlled discovery procedure.
- **imatrix agreement ρ** (pre-registered bar ρ ≥ 0.5) is still uncomputed — a P3 item. It is corroboration, reported either way.
- **Recipe-emission wart:** the `output.weight` pattern (emitted for tied embeddings) regex-matches every `blk.*.attn_output.weight` — harmless here (embed alloc Q5_K), dropped/anchored in P3.

## Why post a null result

Because the measurement machinery is the contribution so far, and it validated: a screen that predicts exact ranking, a bit-cost model that reproduces llama.cpp exactly, and a closed loop that runs end to end and tells the truth when the recipe doesn't win. Whether measured allocation wins on quality is an open empirical question, and the Qwen3-1.7B run will answer it. A forensics tool that can't report its own null result isn't a forensics tool.

## What's next

Phase C (Qwen3-1.7B, post-Oct-17 GPU refresh): the transfer test and the decisive quality test. P3 ship items: imatrix agreement ρ, the `output.weight` emission fix, the real CLI (`bitblame scan/probes/recipe/validate/report`), unit tests, README, git repo, HF dataset + Space.

Full numbers: P2_REPORT.md in the repo (github.com/habib-analyst/bitblame). Method spec: METHODOLOGY.md v1.1.

— Habib Ur Rehman
