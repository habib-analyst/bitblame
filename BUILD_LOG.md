# BitBlame — Build Log

## P0 LOCKED — 2026-10-14 ~03:30 PKT
**Methodology invention phase complete. Three workstreams:**

1. **Gap verification** (`~/workspace/gap-verification/bitblame-gap-report-2026-10-14.md`): blanket novelty claim ("no integrated dynamic tool exists") is FALSE — firefly (torchao/vLLM) and mlx-optiq (MLX, KL+knapsack) exist. Gap HOLDS in narrowed scope: (i) measured-sensitivity recipes for llama.cpp/GGUF (`--tensor-type` overrides + imatrix guidance), (ii) paired-bootstrap CIs on attribution, (iii) token-category blame, (iv) exact-DP K-bit knapsack in GGUF world, (v) closed-loop re-quant driving `llama-quantize`. Positioning: "firefly/mlx-optiq-class dynamic attribution for the llama.cpp/GGUF ecosystem."

2. **Methodology v1.0** (`~/workspace/bitblame/METHODOLOGY.md`, 874 lines, §§0–14): LOLO mock-quant noise scan (unbiased, variance-matched) → exact top-k verification → directional KL metric (mean per-token D_KL, justified via NLL decomposition) → 250-item failure-mode probe battery → BCa 95% paired-bootstrap CIs with Holm correction → multiple-choice knapsack bit-budget optimizer solved by exact DP → closed-loop validation via real `llama-quantize` with pre-registered success criterion. Every section flagged [STANDARD]/[ADAPTED]/[NOVEL]. CPU-first path (Phase A/B) defined; GPU (Phase C, Qwen3-1.7B) deferred to post-2026-10-17 quota refresh.

3. **Tiny CPU prototype** (`prototype.py`, `prototype_results.json`): pure-NumPy 4-layer transformer, per-layer perturbation → KLD ranking validated — 6.3× spread, earlier layers most sensitive, stable top-2 across seeds, 0.1–0.2s CPU. Plumbing proven (caveat: synthetic noise, not real quantization).

**Cross-chat:** RSNA kernel (`rsna-knee-effnet-efficiency-track`) found in ERROR state during read-only env check — alert posted to bounties-competitions inbox; nothing touched. Shared Kaggle GPU quota at 1.52h/30h until 2026-10-17 refresh — BitBlame stays on CPU; their chat notified.

## P1 STARTED — 2026-10-14 ~03:30 PKT
Full pipeline implementation on CPU per METHODOLOGY.md Phase A→B (SmolLM2-135M end-to-end, incl. llama-quantize closed-loop attempt). Target deliverable: P1_REPORT.md with real measured rankings + CIs.

## P2 COMPLETED — 2026-10-14 (final, n=250)
All five P2 gaps closed by the completion worker:
1. Recipe probe arm completed 200/250 → 250/250 via `scripts/resume_recipe_probes.py` (battery IDs verified identical before resume; one service-restart interruption at 225/250, resumed to 250). Recipe accuracy on full battery: 11.6%.
2. Closed-loop harness fixed: verification is now pure-Python GGUF header read (`src/bitblame/gguf.py`) + full type simulation (`optimizer.predict_file_types`); `llama-gguf r` self-test dropped (it asserts data[0]==100.0 on any real file). Re-ran: EXIT 0, 272 tensors, 0 mismatches, `closedloop.json` written; re-quantize deterministic (same sizes).
3. Optimizer bit-cost model v1.1: llama.cpp remap modeled from source (`tensor_type_fallback` demotion, Q4_K_M mixer, mixer-only counters, `--tensor-type` regex first-match incl. the `output.weight` catch-all on mixer-block attn_output). `scripts/verify_recipe_sizes.py` reproduces 272/272 types + bytes exactly (0.0000%) in both GGUFs; old uniform model was −27.3%/−35.0% off. 10.5%-smaller claim re-verified from real files (ratio 0.8949).
4. METHODOLOGY.md bumped to v1.1 (additive §15 changelog): microant + morshoto citations, differentiator (ii) tightened, screen-exact ρ ≥ 0.7 blocking gate, imatrix ρ ≥ 0.5 corroboration bar, DP unit set = gated set, blocking-gates §8.4.
5. P1_REPORT.md now carries a superseded-pointer to P2_REPORT.md.
6. FINAL VERDICT recomputed (`scripts/final_validate.py`, fixed seeds, `hidden_files/phaseB_validation.json`): primary Δ = 0.00 pp (15W/15L/220T), 95% BCa CI [−4.8, +4.0] pp, p = 0.53 → NO_SIGNIFICANT_DIFFERENCE. Accuracies FP16 12.8% / Q4_K_M 11.6% / recipe 11.6%. Compression non-inferiority FAILS (CI low −4.8 < −0.5). Needle "regression" is mechanical: recipe 3/40 = FP16 reference 3/40; baseline 7/40 is luck above reference. Gates: split-half ρ = 0.9754, screen-exact ρ = 0.9286/0.9286/0.7857 — all pass. Still open for P3: imatrix ρ, `output.weight` emission wart, CLI, tests, README, git, HF dataset + Space.
