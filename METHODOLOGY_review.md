# BitBlame — Independent Methodology Review

**Date:** 2026-10-14 ~10:05 PKT · **Reviewer:** Muse (subagent, design-only task)
**Scope:** re-verify the gap claims in METHODOLOGY.md v1.0 (§11) + gap report against LIVE sources; assess whether P0 is truly locked. No code written, nothing published, no other chats' work touched.

## (a) Gap-claim verification (live, 2026-10-14)

**Claim 1 — firefly (neelvad/firefly) is the closest prior art, torchao/vLLM only: CONFIRMED.**
Live GitHub evidence: `quant-sensitivity` quantizes one decoder layer at a time with real torchao kernels and ranks layers by output divergence at final_norm (relative-L1, not KL); `--granularity layer|linear`; `quant-recipe` has isolated/marginal/greedy strategies with accuracy-bar gating and its docs *explicitly scope out the general K-bit case* ("the keep-fp-vs-quantize case is exactly binary subset selection"); closed loop re-evaluates the *served* checkpoint. Backends: HF/torchao, vLLM, SGLang — no GGUF/llama.cpp anywhere; no bootstrap CIs found anywhere in the repo. One respect in which firefly is *ahead* of BitBlame: per-attention-head drill-down and op-level divergence attribution — but on the wrong backend for the GGUF ecosystem.

**Claim 2 — mlx-optiq: KL sensitivity + greedy knapsack, MLX-only: CONFIRMED.**
Live model cards (mlx-community/Qwen3.5-4B-OptiQ-4bit, YOLO26n-OptiQ-6bit): "Per-layer bit-widths come from a KL-divergence sensitivity pass on a six-domain calibration mix... assigns optimal per-layer bit-widths using greedy knapsack optimization"; CLI `optiq convert <hf-model-id> --target-bpw 5.0 --candidate-bits 4,8`. A third-party design doc (joshuarossi/mlx-bun) quotes optiq's exact loop: reference logits → simulate-quantize ONE group (real quantize→dequantize round-trip) → KL on full output distribution → greedy knapsack. So mlx-optiq's *sensitivity measurement is more faithful than BitBlame Stage 1* (exact round-trip vs uniform-noise screen); BitBlame's answer is cheaper screening + Stage-2 exact verification. No CIs, no token-category blame, no documented closed loop, Apple-Silicon/MLX only.

**Claim 3 — llama.cpp per-layer rules are unmeasured folklore: CONFIRMED.**
The gap report read `src/llama-quant.cpp` directly (use_more_bits at L777–781 = first 1/8 + last 1/8 + every-3rd-layer, verified in source). Only empirical counter-evidence is a one-model demo (AmberLJC, Qwen2-0.5B, PPL-only, Q4_0/Q8_0) — no general measurement program exists.

**Claim 4 — two quantbenches, neither attributing: CORRECTION.**
The gap report verified stef41/quantbench is static-only (header parsers). A *second* repo, jamespa16/quantbench, DOES execute the model (llama-server + HumanEval+ pass@1) but is aggregate-only: one score per quant, CSV + bar chart, no layer signal whatsoever. The claim "quantbench never executes" holds only for stef41's. Neither affects BitBlame's gap.

**New find NOT in the gap report — microant (Code-byte404/microant, 5 stars, MIT, created 2026-07-10): whole-model GGUF/MLX capability ruler with bootstrap CIs.**
It runs a 250-case capability pack across quant tiers (GGUF Q2_K/Q3_K_M ladders included), reports per-tier deltas with bootstrap 95% CIs, and its `confirm` stage closes the loop: a rung counts as held only if the test CI's *lower bound* clears the bar — methodologically aligned with BitBlame §8. Its "within-chain failure attribution" is tier/chain damage breakdown, NOT per-layer blame; it does no sensitivity measurement, no bit allocation, no recipe emission; actual model runs need Apple Silicon. Impact: it erodes the loose reading of differentiator (ii) ("bootstrap CIs + GGUF") but leaves the precise claim ("CIs wired into per-layer attribution feeding allocation") untouched. §11 must cite it and tighten (ii)'s wording.

**Related sibling — morshoto/qwen-meetup-tokyo exp_002:** paired-bootstrap practical-equivalence (Q4_K_M vs Q8_0 GGUF, ±10pp margin at 95%). Whole-model comparison only; no attribution. Cite as a methodological sibling for §8, not a threat.

**Differentiator-by-differentiator verdict (live):**

| # | Differentiator | Live status |
|---|---|---|
| (i) measured-sensitivity GGUF recipes via `--tensor-type` + imatrix guidance | HOLDS — nothing found emits measured per-tensor recipes for llama.cpp; systematic-adaptive-layer-quantization is 0-star research scripts with activation-variance proxy, not causal |
| (ii) paired-bootstrap CIs wired into attribution→allocation | HOLDS but needs rewording — CIs on GGUF *measurement* exist (microant, morshoto); CIs on *per-layer blame feeding the optimizer* do not |
| (iii) token-category blame | HOLDS — firefly has layer/head granularity, mlx-optiq layer granularity; nobody does position-bucket/outlier-token KL breakdowns with CIs |
| (iv) exact-DP K-bit knapsack in the GGUF world | HOLDS — firefly explicitly scopes out K-bit; optiq's knapsack is MLX-only and 2-choice (4,8); HAWQ-V3's ILP needs a solver dependency |
| (v) closed-loop re-quant driving llama-quantize | HOLDS — firefly proves the pattern for torchao; microant's confirm is whole-model; nobody drives `llama-quantize` in the loop |

## (b) Corrections and missing pieces (section-specific)

1. **§11 / gap report — missing citation:** add microant and morshoto exp_002 as measurement-layer prior art; reword differentiator (ii) to "paired-bootstrap CIs on *per-layer attribution, significance-gating the allocation*" (per §11.5 correction policy).
2. **§2 (Stage 1 vs Stage 2) — add a screen-exact agreement gate.** mlx-optiq measures sensitivity with exact quantize→dequantize round-trips; BitBlame's noise screen is deliberately less faithful. The spec never requires reporting how well the screen predicts the exact verification. Patch: Stage 2 must report Spearman ρ between Stage-1 S_u ranking and Stage-2 exact KL ranking for the top-k; if ρ < 0.7, the screen is unreliable and k grows (or the mock model is revised). This is the single most load-bearing methodological check and it is currently implicit.
3. **§7.4.2 (imatrix agreement diagnostic) — underspecified.** "Agreement ⇒ corroboration" has no threshold. Patch: pre-register Spearman ρ ≥ 0.5 between imatrix per-tensor importance and S_u as the corroboration bar; report the number either way.
4. **§6.4 item 2 — the significance gate pins non-gated units at 4 bits.** This interacts with the knapsack: the DP (§7.3) optimizes over all units but the gate pins most. Make explicit that the DP's unit set IS the gated set 𝒰′ (§7.1 already says this — the fix is one sentence in §7.3, not a design change).
5. **P1_REPORT.md is mostly TBD** (Phase A section blank, all result tables blank, `llama.cpp rev TBD` in §0). The `llama.cpp` revision is load-bearing for differentiator (i) (`--tensor-type` regex syntax must be pinned against a recorded revision). Fill before any P2 claim about recipe validity.

## (c) Verdict

**The narrowed positioning HOLDS after independent live re-verification** — firefly/mlx-optiq-class dynamic attribution *for the llama.cpp/GGUF ecosystem*, with KL sensitivity + paired-bootstrap CIs + token-category blame. The blanket "first integrated tool" claim stays retracted, correctly.

**P0 is design-locked, not evidence-locked.** METHODOLOGY.md v1.0 is fully specified, falsifiable, and correctly scoped — "locked" as a design contract. But the empirical anchor under the lock is thin: the prototype is a 4-layer *random* transformer with *synthetic* noise (proves plumbing, proves nothing about real quantizers), and P1_REPORT.md's TBDs are the actual P0→P2 bridge. **Do not proceed to P2 (GPU-scale recipe validation) until:** (1) P1 fills the TBDs, especially the real `llama-quantize` closed loop on SmolLM2-135M; (2) the Stage-2 screen-exact agreement gate (correction #2) is measured and reported; (3) the split-half reliability gate (§2.3, ρ ≥ 0.9) passes — P1 runs at N=16/T=256 (below the spec's N=32/T=512), so this gate is the load-bearing item; a failed gate means growing N, not trusting the ranking.

*Required spec patches: correction policy §11.5 covers them — bump spec to v1.1 with microant citation + reworded (ii), the Stage-2 agreement gate, and the imatrix ρ ≥ 0.5 bar.*
