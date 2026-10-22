DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

r/MachineLearning post — BitBlame: dynamic per-layer quantization attribution for GGUF with paired-bootstrap CIs

Title: [P] BitBlame: dynamic per-layer quantization attribution for GGUF/llama.cpp with paired-bootstrap CIs — screen validated (ρ = 0.9286), first recipe a null result

---

BitBlame is a forensics tool that localizes quantization sensitivity per layer (unit) for the llama.cpp/GGUF ecosystem, with uncertainty quantification wired into the allocation decision. Method spec (v1.1) and full Phase B report are in the repo (github.com/habib-analyst/bitblame).

Method (two-stage design — never pay for information twice):

1. **Screen (Stage 1):** LOLO mock-noise perturbation. Each unit's weights get uniform noise matched to quantization variance (per-group scale, k-quant super-block grouping), scored by mean per-token KL divergence against the FP16 reference. R = 2 draws, BCa 95% CIs (B = 2000) over calibration sequences, null-floor run (8-bit noise) as a significance gate, split-half reliability gate (ρ ≥ 0.9).
2. **Exact verification (Stage 2, blocking gate ρ ≥ 0.7):** top-8 units re-measured with faithful quantize→dequantize at 3/4/6 bits, per-unit loss-vs-bit-width slopes fit and shrunk toward the theoretical α = 2 ln 2 ≈ 1.386.
3. **Allocation (Stage 4):** significance-gated exact-DP multiple-choice knapsack over 6 real GGUF quant types → `--tensor-type` overrides + imatrix guidance. Non-gated units are pinned at the default; the DP optimizes only over the gated set.
4. **Validation (Stage 5):** re-quantize with real `llama-quantize`, re-run a 250-item paired probe battery (MMLU/GSM8K/HumanEval/synthetic needle/copy), pre-registered paired-bootstrap test (BCa, B = 10,000).

Phase B results (SmolLM2-135M, CPU, n = 250):

- **Screen validity (the top technical risk):** screen-vs-exact Spearman ρ = 0.9286 at 4-bit (p = 0.0009, n = 8), ρ = 0.9286 at 3-bit, ρ = 0.7857 at 6-bit. Gate passes. Split-half rank stability ρ = 0.9754. 31/31 units pass the significance gate (null-floor max CI-high = 0.00307).
- **Concentration:** top-3 units (embed + blk.29 + blk.11) = 57.9% of total measured sensitivity; embedding alone 32.0% (S = 0.3674, 95% CI [0.3471, 0.3927]).
- **Optimizer fidelity:** the llama.cpp-remap-aware bit-cost model (K-quant fallback demotion, Q4_K_M mixer, `--tensor-type` regex first-match) reproduces 272/272 tensor types and stored bytes exactly (0.0000% error) on both GGUFs. Closed-loop harness exit 0, 272 tensors verified, 0 mismatches.
- **Recipe outcome:** 10.5% smaller than uniform Q4_K_M (94,373,920 B vs 105,453,856 B, ratio 0.8949); accuracy 11.60% = 11.60%; Δ 0.00 pp; 95% BCa CI [−4.8 pp, +4.0 pp]; p = 0.5338; 15 wins / 15 losses / 220 ties.

[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] The quality-win claim is NOT established: verdict NO_SIGNIFICANT_DIFFERENCE, and non-inferiority at a 0.5 pp margin FAILS (CI too wide). The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

Prior art (honest accounting): firefly (torchao, real kernels, binary allocation, no CIs, no GGUF); mlx-optiq (KL sensitivity, greedy knapsack, MLX only, no CIs, no documented closed loop); microant (whole-model GGUF capability ruler with bootstrap CIs — no per-layer blame, no allocation, no recipes); morshoto exp_002 (paired-bootstrap practical equivalence, whole-model only); quantbench variants (static header parsing / aggregate-only execution). BitBlame's precise differentiators: GGUF/llama.cpp recipes; paired-bootstrap CIs on per-layer attribution, significance-gating the allocation (CIs on whole-model GGUF measurement exist — the novel claim is CIs on per-layer blame feeding the allocator); token-category blame; exact-DP knapsack over 6 GGUF types; closed-loop validation through real `llama-quantize`.

Limitations: SmolLM2-135M only — no transfer claim to larger models. The battery is underpowered at this scale (FP16 itself 12.80%; GSM8K 1.7–3.3%, HumanEval 0%, copy 0%), so the verdict is "no evidence of a difference," not "evidence of no difference." Calibration 16 seqs × 256 tokens (spec wants 32×512 — CPU adaptation). Greedy decoding only, English calibration only. Per-unit CIs are marginal, not family-wise. On needle the recipe matched the FP16 reference count (3/40); the baseline scored above the reference by chance — not a regression. imatrix agreement ρ (bar ρ ≥ 0.5) still uncomputed.

I am posting the null result deliberately: the measurement machinery (validated screen, exact optimizer, closed loop) is the contribution so far; whether measured allocation wins on quality is an open empirical question Phase C will answer.
