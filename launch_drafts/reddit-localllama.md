DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

r/LocalLLaMA post — BitBlame: per-layer quantization blame for GGUF, with an honest null result

Title: BitBlame: per-layer quantization blame for GGUF models, with CIs — and an honest null result on the first recipe

---

If you quantize GGUF models with llama.cpp, the quantizer allocates bits by heuristics — nothing measures which layers actually take the damage. I built BitBlame to measure that: per-layer (per-unit) sensitivity for the llama.cpp/GGUF ecosystem, with bootstrap CIs on every number.

How it works:

- Stage 1: perturb one unit at a time with quantization-calibrated mock noise, measure mean per-token KL vs the FP16 reference over calibration sequences. Unit = one transformer block (embedding is its own unit).
- Stage 2: re-measure the top 8 units with exact quantize→dequantize at 3/4/6 bits.
- Stage 4: significance-gated exact-DP knapsack over 6 real GGUF quant types → `--tensor-type` overrides for `llama-quantize` + imatrix guidance.
- Stage 5: closed loop — re-quantize with real `llama-quantize`, re-run a 250-item probe battery, pre-registered paired-bootstrap verdict.

What held up on SmolLM2-135M (CPU, full pipeline):

- The cheap screen predicts the exact ranking: Spearman ρ = 0.9286 on the top 8 (p = 0.0009, n = 8). Split-half rank stability ρ = 0.9754. All 31 units pass the significance gate (null floor max CI-high = 0.00307).
- Concentration: top-3 units (embedding + blk.29 + blk.11) = 57.9% of measured sensitivity; embedding alone 32.0%.
- The loop works end-to-end: harness exit 0, 272/272 tensor types verified, and the optimizer's bit-cost model reproduces both GGUFs' tensor types and stored bytes exactly (0.0000% error) — it models llama.cpp's K-quant fallback demotion, the Q4_K_M mixer, and `--tensor-type` first-match semantics.

The honest part:

The recipe came out 10.5% smaller than uniform Q4_K_M (94,373,920 B vs 105,453,856 B) and scored exactly the same on the full 250-item battery: 11.60% = 11.60%, Δ 0.00 pp, 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338.

[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] The quality-win claim is NOT established: verdict NO_SIGNIFICANT_DIFFERENCE, and non-inferiority at a 0.5 pp margin FAILS (CI too wide). The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

Related work, honestly: this is firefly/mlx-optiq-class attribution for GGUF. firefly does per-layer attribution with real torchao kernels (no GGUF, no CIs); mlx-optiq does KL sensitivity + greedy knapsack (MLX only, no CIs, no closed loop); microant is the closest adjacent work (whole-model GGUF capability ruler with bootstrap CIs, but no per-layer blame, no allocation, no recipes). BitBlame's differentiators: GGUF/llama.cpp recipes, paired-bootstrap CIs on per-layer attribution significance-gating the allocation, token-category blame, exact-DP knapsack, closed-loop validation.

Limits: SmolLM2-135M only, no transfer claim. Battery near the floor (FP16 itself 12.80%, HumanEval 0% for everyone). Greedy decoding, English calibration. One more detail for the record: on needle, the recipe matched the FP16 reference's count (3/40) — the baseline scored above the reference by chance, so this is not a retrieval regression.

Code + full report (P2_REPORT.md with every number) coming to github.com/habib-analyst/bitblame. Happy to answer technical questions — especially challenges to the mock-noise screen, which is the part I'd most like scrutinized.
