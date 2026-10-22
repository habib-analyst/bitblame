DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

LinkedIn post — BitBlame

---

Quantization doesn't break models evenly. Usually a few layers take most of the damage, and the quantizer never measures which ones.

I built BitBlame to measure it: per-layer quantization blame for GGUF / llama.cpp models, with a confidence interval on every number.

On SmolLM2-135M, the full pipeline ran end to end on CPU:

- The cheap screen predicts the exact ranking: Spearman ρ = 0.9286 on the top 8 units, split-half rank stability ρ = 0.9754.
- Concentration: the top 3 units (embedding + blocks 29 and 11) carry 57.9% of measured sensitivity. The embedding alone is 32.0%.
- The closed loop works through real llama-quantize: 272/272 tensor types verified, and the optimizer's bit-cost model reproduces both GGUFs exactly (0.0000% error).

And the honest part: the first recipe came out 10.5% smaller than uniform Q4_K_M and scored exactly the same on the full 250-item battery, 11.60% = 11.60% (Δ 0.00 pp, 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338).

[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] The quality-win claim is NOT established: verdict NO_SIGNIFICANT_DIFFERENCE, and non-inferiority at a 0.5 pp margin FAILS because the interval is too wide. The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

What this is: firefly/mlx-optiq-class attribution for the llama.cpp/GGUF ecosystem, with paired-bootstrap CIs on per-layer blame, token-category blame, an exact-DP knapsack over real GGUF quant types, and closed-loop validation. What it is not: the first such tool, or a proven quality win. Not yet.

Limits, stated plainly: 135M only, no transfer claim. The battery sits near the floor (FP16 itself 12.80%). Greedy decoding, English calibration only.

I am sharing the null result on purpose. The measurement machinery is validated; whether measured allocation wins on quality is an open question, and the Qwen3-1.7B run will answer it.

Full numbers in P2_REPORT.md. Repo: github.com/habib-analyst/bitblame

— Habib Ur Rehman
