DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

X thread — BitBlame (10 posts)

1/10
Quantization doesn't break models evenly. One layer usually takes most of the damage, and nobody measures which one. I built BitBlame to measure it: per-layer blame for GGUF models, with confidence intervals. Honest results below, including the part that didn't work.

2/10
How it works: perturb one layer at a time with quantization-calibrated noise, measure how far the output distribution moves (mean per-token KL vs FP16). Rank layers. Verify the top 8 with exact quantize-dequantize. Allocate bits with an exact-DP knapsack.

3/10
First, is the cheap screen trustworthy? Screen vs exact ranking: Spearman ρ = 0.9286 on the top 8 (p = 0.0009, n = 8). Split-half rank stability ρ = 0.9754. All 31 units pass the significance gate. The screen predicts the exact ranking on this model.

4/10
The concentration finding: on SmolLM2-135M, the top 3 units (embedding + blocks 29 and 11) carry 57.9% of measured sensitivity. The embedding alone is 32.0%. If you protect 3 units, you cover most of the measured damage.

5/10
The loop is real: blame → allocation → re-quantize with actual llama-quantize → re-measure → paired-bootstrap verdict. Harness exits 0, 272/272 tensor types verified. The optimizer's bit-cost model reproduces both GGUFs exactly (0.0000% error).

6/10
The recipe came out 10.5% smaller than uniform Q4_K_M (94,373,920 B vs 105,453,856 B). And on the full 250-item battery it scored... exactly the same. 11.60% = 11.60%. Δ 0.00 pp, 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338.

7/10
[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] Quality win NOT established: NO_SIGNIFICANT_DIFFERENCE, non-inferiority at 0.5 pp FAILS (CI too wide). Same quality at 10.5% smaller on one 135M model, not a win. Decisive Qwen3-1.7B test pending GPU quota (Oct 17).

8/10
Context: this is firefly/mlx-optiq-class attribution for the llama.cpp/GGUF ecosystem. Differentiators: GGUF recipes, paired-bootstrap CIs on per-layer blame gating allocation, token-category blame, exact-DP knapsack, closed-loop validation. Not the first such tool, and I won't claim it is.

9/10
Limits, stated plainly: SmolLM2-135M only, no transfer claim. Battery near the floor (FP16 itself 12.80%, HumanEval 0%). Greedy decoding, English calibration. "No evidence of a difference" is not "evidence of no difference."

10/10
BitBlame is a measurement tool with validated attribution and an honest null result on its first recipe. Code + full report (P2_REPORT.md) coming to github.com/habib-analyst/bitblame. The Qwen3-1.7B run decides what this becomes.
