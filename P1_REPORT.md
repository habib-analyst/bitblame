# BitBlame P1 Report — Phase A → Phase B (CPU)

> **SUPERSEDED — do not cite.** The canonical results record is
> [`P2_REPORT.md`](P2_REPORT.md) (final verdict, recomputed from raw
> artifacts). This P1 draft's TBD placeholders were filled there; the
> tables below are stale/incomplete and are kept only as build history.

**Date:** 2026-10-14 · **Spec:** METHODOLOGY.md v1.0 · **Model:** SmolLM2-135M (trained, 134.5M params, 30 layers) · **Compute:** 2-CPU VM, torch 2.14.1+cpu (fp16), llama.cpp rev `TBD`

> Every number below is a real measured value from the runs described.
> Nothing is invented or placeholder. Where the run deviated from the
> spec (all CPU-time adaptations), the deviation is stated explicitly.

## 0. Environment

| item | value |
|---|---|
| torch | 2.14.1+cpu |
| transformers / datasets | 5.19.0 / 5.1.0 |
| numpy / scipy / pandas / matplotlib | 2.5.4 / 1.18.1 / 3.0.6 / 3.11.2 |
| model | HuggingFaceTB/SmolLM2-135M, safetensors, loaded fp16 on CPU |
| calibration | FineWeb-Edu sample/10BT/000_00000.parquet (row group 0 via HTTP range requests; 60 docs cached), N=TBD × T=TBD, neutral prose, >5% non-ASCII / code-fence filtered |
| units | 31 = 30 transformer blocks (Linear weights) + embed (lm_head tied to embed_tokens — single unit) |

## 1. Phase A — plumbing (tiny random transformer, CPU)

*TBD — will summarize the green run.*

## 2. Stage 1 — LOLO mock-noise scan (b=4, R=2)

### 2.1 Per-layer sensitivity ranking (blame score S_u = mean token KL, nats)

Top 10 and bottom 5 with 95% BCa CIs (B=2000, joint resampling):

*TBD table*

### 2.2 Reliability gates

*TBD: null floor, split-half rho, draw-to-draw CV flags*

## 3. Token-category breakdown (§4.2)

*TBD: position buckets, outlier tokens, with CIs*

## 4. Stage 2 — exact top-k verification + slope calibration

*TBD: KL at b=3/4/6, fitted alpha vs theory 1.386*

## 5. Bit-budget optimizer → recipe.sh

*TBD: allocation, predicted loss, budget*

## 6. Stage 3/5 — closed loop through llama.cpp

*TBD: imatrix, quantize, tensor-type verification, probe results, paired bootstrap verdict*

## 7. Runtime numbers (wall clock, this VM)

| stage | time |
|---|---|
| env setup (pip + downloads) | TBD |
| llama.cpp build (4 tools) | TBD |
| HF→GGUF conversion | TBD |
| Stage 0/1 scan (31u × 2 draws) | TBD |
| null-floor run | TBD |
| bootstrap + categories | TBD |
| Stage 2 exact (8u × 3bw) | TBD |
| optimizer + recipe | TBD |
| imatrix + 2× quantize | TBD |
| probe battery × 3 variants | TBD |

## 8. What worked

*TBD*

## 9. What didn't / blockers

*TBD*

## 10. Honest caveats

1. Mock-noise screen is approximate (§2.1 limitations a–c); Stage 2 buys back exactness only for the top-k.
2. N=16/T=256 calibration (spec §12: 32×512) — CPU-time adaptation; split-half gate decides sufficiency.
3. Tiny-model transfer (§13.7): 135M rankings are not claimed to transfer to 1.7B — Phase C work.
4. Per-unit CIs are marginal, not family-wise (§6.3 honesty clause).
5. Greedy decoding only; English calibration only.
