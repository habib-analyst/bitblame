DRAFT — DO NOT POST — pending Phase C (Qwen3-1.7B)

HN Show HN — BitBlame: per-layer quantization blame for GGUF models, with confidence intervals

Title: Show HN: BitBlame – per-layer quantization blame for GGUF models, with CIs

---

BitBlame measures which layers of a model take the damage when it is quantized, for the llama.cpp/GGUF ecosystem — with bootstrap confidence intervals on every blame number, and a closed loop that turns the measurement into a mixed-precision recipe and re-validates it through real `llama-quantize`.

Pipeline: mock-noise LOLO screen (per-unit mean token KL vs FP16) → exact verification of the top 8 units with real quantize→dequantize → significance-gated exact-DP knapsack over 6 GGUF quant types → `--tensor-type` recipe → re-quantize, re-measure, pre-registered paired-bootstrap verdict.

Validated so far on SmolLM2-135M (CPU, full pipeline):

- Screen vs exact ranking: Spearman ρ = 0.9286 on the top 8 (p = 0.0009, n = 8). Split-half rank stability ρ = 0.9754. 31/31 units pass the significance gate.
- Concentration: top-3 units (embedding + blocks 29 and 11) = 57.9% of measured sensitivity; embedding alone 32.0%.
- Optimizer bit-cost model reproduces both GGUFs' 272/272 tensor types and stored bytes exactly (0.0000% error). Closed-loop harness exit 0.
- Recipe: 10.5% smaller than uniform Q4_K_M (94,373,920 B vs 105,453,856 B), and scored exactly the same on the full 250-item battery: 11.60% = 11.60%, Δ 0.00 pp, 95% BCa CI [−4.8 pp, +4.0 pp], p = 0.5338.

[PHASE-C-DEPENDENT — REVISE AFTER PHASE C] The quality-win claim is NOT established: verdict NO_SIGNIFICANT_DIFFERENCE, and non-inferiority at a 0.5 pp margin FAILS (CI too wide). The honest reading is same quality at 10.5% smaller on this one small model, not a demonstrated quality win. The decisive transfer test on Qwen3-1.7B (Phase C) is pending a Kaggle GPU quota refresh (Oct 17).

Positioning, honestly: firefly/mlx-optiq-class attribution for GGUF. firefly (torchao, no GGUF, no CIs), mlx-optiq (MLX only, no CIs, no closed loop), microant (whole-model GGUF ruler with CIs, no per-layer blame, no recipes). Differentiators: GGUF recipes, paired-bootstrap CIs on per-layer blame gating allocation, token-category blame, exact-DP knapsack, closed-loop validation.

Limits: 135M only, no transfer claim; battery near the floor (FP16 itself 12.80%); greedy decoding, English calibration only. Full numbers in P2_REPORT.md. Repo: github.com/habib-analyst/bitblame.

The part I'd most like scrutinized: the mock-noise screen. It validated on the top 8 here — is that enough to trust it on a 1.7B model, or does the screen need exact verification on more units before allocation?
