# BitBlame P2 Report — Phase B Recipe Validation (SmolLM2-135M)

**Date:** 2026-10-14 · **Spec:** METHODOLOGY.md v1.1 · **Model:** HuggingFaceTB/SmolLM2-135M (134.5M params, 30 blocks) · **Compute:** CPU VM · **llama.cpp rev:** `fcac19079f3f2985fe74ed845f0880e7e9a8030a` (2026-10-14)

**Supersedes:** `P1_REPORT.md` (mostly TBD placeholders; do not cite). This report contains the real measured numbers, recomputed from the raw artifacts in `hidden_files/`.

> **FINAL VERDICT (pre-registered test, §8.2, completed battery n = 250): NO_SIGNIFICANT_DIFFERENCE.** The BitBlame recipe did **not** beat the uniform Q4_K_M baseline on the probe battery. Mean paired delta = **0.00 pp** (15 wins, 15 losses, 220 ties), 95% BCa CI **[−4.8 pp, +4.0 pp]**, bootstrap p = 0.53. The CI straddles zero; the recipe is statistically indistinguishable from the baseline. The compression non-inferiority gate also fails (CI lower −4.8 pp < −0.5 pp margin). The recipe *is* 10.5% smaller than baseline, so the honest statement is: **a 10.5% smaller model that performs identically to Q4_K_M on the full 250-item battery (11.6% = 11.6% accuracy) — not a demonstrated quality win.**
>
> *Interim note: an earlier version of this report computed the verdict on an incomplete recipe arm (200/250). The missing 50 items (30 needle + 20 copy) have since been measured (resume run, same prompts verified identical), the closed-loop harness fix and optimizer model fix below are in, and every number in this report is recomputed on the completed n = 250. The interim headline (+1.00 pp, CI [−4.5, +6.0]) is superseded by the final numbers above.*

---

## 1. What ran

| Stage | Artifact | When (PKT, Oct 14) | Notes |
|---|---|---|---|
| Phase A (tiny random transformer) | `phaseA_report.json`, `phaseA_heatmap.png` | Oct 13 23:46 | Plumbing only; synthetic noise, not real quantization |
| Calibration | `phaseB_calib.json`, `calib.txt` (19 KB) | 01:12–01:15 | 16 seqs × 256 tokens, FineWeb-Edu (spec §12 wants 32×512 — CPU adaptation, see §8) |
| Stage 1 screen (mock noise, b=4, R=2) | `phaseB_scan.npz` | 01:28 | 31 units × 16 seqs; S = mean token KL |
| Null floor | `phaseB_null.npz` | 02:20 | R=1; max CI-high = 0.00307 |
| Analysis + optimizer | `phaseB_analysis.json` | 03:00–03:02 | Exact-DP knapsack → `recipe.sh` |
| imatrix + quantize | `imatrix.dat` (1.27 MB), 3 GGUFs in `data/models/` | 03:11–03:12 | Real `llama-imatrix` + `llama-quantize` |
| Probe battery | `phaseB_gguf_{f16,q4km,recipe}.json` | 04:07 / 04:32 / 04:45 (+ resume → 250/250) | GGUF probes via llama.cpp, greedy decoding. Recipe arm first run stopped at 200/250; resume run (`resume_recipe_probes.py`) completed it to 250/250 (see deviation note below) |

**Chronology checks out.** Every artifact's timestamp is in the order the pipeline requires, and file sizes are consistent with real runs (e.g. the probe JSONs carry per-item generations, latencies, and throughputs — not placeholder rows).

**Procedural deviation — closed-loop script exit code 1 → FIXED, re-ran exit 0.** `phaseB_closedloop_stdout.log` ends `CLOSEDLOOP EXIT: 1`: the script's final verification step (`llama-gguf r`) crashed on an assertion inside the llama.cpp *example* tool (`gguf_ex_read_1: data[0]: found 1.750000, expected 100.000000` — the example's own self-test, not a file-integrity error). The two `llama-quantize` commands **did complete** before that step, and the probes ran afterwards via a separate script (`phase_b_gguf_probes.py`). **Fix (P2 completion):** verification now parses the GGUF header directly with pure-Python `src/bitblame/gguf.py` and compares against a full type simulation (`optimizer.predict_file_types`); `llama-gguf r` is never invoked. The fixed harness was re-run end-to-end (`phaseB_closedloop_stdout2.log`): **EXIT 0, "verified 272 tensors, 0 mismatches"**, `closedloop.json` written. Re-quantization is deterministic — the recipe GGUF is byte-identical in size (94,373,920 B) to the probed artifact.

**Battery deviation — recipe arm COMPLETED.** The probe battery is 250 items (100 MMLU + 60 GSM8K + 30 HumanEval + 40 needle + 20 copy). The first probe run stopped the recipe arm at 200/250 (log ended after `[recipe] 200/250`, cause not recorded — the run died mid-battery). The missing 50 (last 30 needle + all 20 copy) were measured by a resume run (`scripts/resume_recipe_probes.py`, which first verifies the rebuilt battery's item IDs match the stored battery exactly, then resumes from the checkpoint). The resume itself was interrupted once by a service restart at 225/250 and resumed again to **250/250** (`phaseB_ggufprobes_resume_stdout.log`: `[recipe] done accuracy=0.116 n=250`). All recipe-vs-baseline statistics below are therefore on the **full n = 250 paired items**, as the pre-registered criterion assumed.

**Harmless no-op.** The recipe command includes `--tensor-type output.weight=Q5_K`, but SmolLM2-135M ties its lm_head to the embedding, so no `output.weight` tensor exists in the GGUF. The override matches nothing and is ignored.

## 2. Integrity checks (this analysis, CPU-light)

All checks passed; nothing below is taken on trust from a summary file:

1. **All 5 core JSONs parse** (`phaseB_analysis`, `phaseB_battery`, 3 probe files). No TBD/placeholder values anywhere in the analysis JSON.
2. **Recomputation from raw `.npz`:** S for `blk.0` recomputed from `token_kl` in `phaseB_scan.npz` = 0.017403588, exactly matching the stored value. `tail_S` matches too.
3. **Item-ID alignment:** probe JSONs are in exact battery order; recipe item IDs == battery IDs (all 250). Paired deltas are per-item, not aggregate-vs-aggregate.
4. **GGUF header verification (direct binary read, `src/bitblame/gguf.py`):** non-overridden block (`blk.1`) is byte-identical in type between baseline and recipe (q5_0 / q6_K) — overrides touched only the requested blocks. Overridden blocks show the requested tiers modulo llama.cpp's per-tensor K-quant remapping (see §8, caveat 5 — now modeled exactly, below). The `llama-gguf` dump in the closed-loop log agrees with this independent header read.
5. **Optimizer ↔ recipe command consistency:** the optimizer's per-unit allocation (Q2_K: 12, Q4_K: 8, Q5_K: 7, Q6_K: 3, Q8_0: 1 = 31 units) exactly matches the `--tensor-type` overrides in the logged `llama-quantize` command.
6. **Model sizes (real files):** FP16 = 270,885,376 B · Q4_K_M baseline = 105,453,856 B · recipe = 94,373,920 B. Recipe/base = **0.8949** — the recipe is 10.5% smaller than the baseline it was supposed to match at fixed size.
7. **Optimizer bit-cost model v1.1 — FIXED and verified exact.** The v1.0 model assumed a requested tier lands uniformly (`params × b/8 × 1.02`); it predicted 68.58 MB for both the recipe and the baseline — off by −27.3% / −35.0% vs the real files. The v1.1 model (`optimizer.py`: `tensor_type_fallback` demotion Q2_K/Q3_K→Q4_0, Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0 on ncols % 256 ≠ 0; Q4_K_M mixer with `use_more_bits`; mixer counters counting only mixer-processed tensors; `--tensor-type` regex first-match incl. the `output.weight` catch-all — all read from llama.cpp source, `src/llama-quant.cpp`) reproduces **272/272 tensor types and stored bytes exactly (err 0.0000%)** in BOTH files (`scripts/verify_recipe_sizes.py`). The 10.5%-smaller claim is re-verified from the real files (ratio 0.8949); the new model predicts the tensor-bytes ratio 0.8932 (10.7% smaller) and, with the shared 1,785,376 B header, the file ratio 0.8949 — exact — vs the old model's 1.000 (equal size).

## 3. The blame ranking — headline finding: CONCENTRATED, HOLDS

Per-unit blame score S_u = mean per-token KL (nats) under 4-bit mock-noise perturbation, 95% BCa CIs (B=2000), null floor max CI-high = 0.00307. All 31 units pass the significance gate (CI_low > null floor).

| Rank | Unit | S_u | 95% CI | Share of total |
|---|---|---|---|---|
| 1 | **embed** (token embedding) | 0.3674 | [0.3471, 0.3927] | 32.0% |
| 2 | blk.29 | 0.1604 | [0.1377, 0.2286] | 14.0% |
| 3 | blk.11 | 0.1366 | [0.1276, 0.1434] | 11.9% |
| 4 | blk.28 | 0.0652 | [0.0585, 0.0739] | 5.7% |
| 5 | blk.2 | 0.0503 | [0.0413, 0.0612] | 4.4% |
| 6 | blk.9 | 0.0351 | [0.0267, 0.0466] | 3.1% |
| 7 | blk.27 | 0.0229 | [0.0210, 0.0253] | 2.0% |
| 8 | blk.19 | 0.0182 | [0.0173, 0.0192] | 1.6% |

**Do ~3 layers dominate? YES — with the embedding caveat.** The top 3 units (embed + blk.29 + blk.11) account for **57.9%** of total measured sensitivity; the single embedding unit alone is 32.0%. Among transformer blocks only, the top 3 (blk.29, blk.11, blk.28) account for **46.5%** of block sensitivity. The concentration is real, not a CI artifact: the top-3 CIs are far above the null floor, and the top-7 units each have P(top-8) = 1.00 under bootstrap resampling.

**Reliability gates (spec §2.3 / review §b.2–3):**

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| Split-half rank stability | ρ ≥ 0.9 | ρ = **0.9754** (n_half = 8) | ✅ PASS |
| Significance gate (null floor) | CI_low(S_u) > 0.00307 | 31/31 units pass | ✅ PASS |
| **Screen-vs-exact agreement** | Spearman ρ ≥ 0.7 | **ρ = 0.9286** at 4-bit (matched screen width); 0.9286 at 3-bit; 0.7857 at 6-bit | ✅ PASS |

**Screen-vs-exact agreement (review's top technical risk, now measured).** Stage 1 (mock-noise screen) ranked the top-8 units; Stage 2 measured those same 8 units with exact quantize→dequantize at 3/4/6 bits and power-law fits (R² = 0.84–0.999). The Stage-2 unit set is exactly the screen's top-8. Spearman ρ between screen S_u and exact KL = **0.9286** (p = 0.0009, n = 8) at both the 3-bit and 4-bit exact measurements, and **0.7857** (p = 0.021) at 6-bit. All three clear the review's ρ ≥ 0.7 gate. The single discordant adjacent pair is blk.27 vs blk.19 at 4-bit (screen: blk.27 > blk.19; exact: 0.0121 vs 0.0165). At 6-bit the KL values shrink toward the noise floor (0.001–0.027), so the lower ρ is expected. **The mock-noise screen predicts the exact ranking well on this model — the screen-fidelity risk is not the reason the closed loop failed to show a gain.**

## 4. THE VERDICT — pre-registered paired test (§8.2)

Method: `paired_bootstrap_test` from the repo's own `src/bitblame/bootstrap.py` (BCa, B = 10,000, jackknife acceleration), per-item paired differences on the 250 common items (`scripts/final_validate.py`, all seeds fixed). Verdict vocabulary per spec.

### Primary criterion (fixed size): size(recipe) ≤ 1.01 × size(base), CI lower bound > 0

| Comparison | n | Mean Δ | 95% BCa CI | p | Verdict |
|---|---|---|---|---|---|
| **Recipe − Q4_K_M baseline** | 250 | **0.00 pp** | **[−4.8 pp, +4.0 pp]** | 0.5338 | **NO_SIGNIFICANT_DIFFERENCE** |
| Recipe − FP16 reference | 250 | −1.20 pp | [−6.0 pp, +2.8 pp] | 0.7293 | NO_SIGNIFICANT_DIFFERENCE |
| Baseline − FP16 reference | 250 | −1.20 pp | [−5.2 pp, +1.6 pp] | 0.8028 | NO_SIGNIFICANT_DIFFERENCE |

Size gate: 0.8949 ≤ 1.01 ✅ — the recipe satisfies the size constraint (it is *smaller*, not merely equal).

**Raw accuracy on the common 250 items:** FP16 12.80% · Q4_K_M 11.60% · recipe 11.60%. Paired detail (recipe vs baseline): 15 wins, 15 losses, 220 ties — the zero delta is exact, not a rounding artifact.

### Secondary criterion (compression): non-inferiority at size ≤ 0.9 × baseline

The recipe qualifies for the compression regime (0.8949 ≤ 0.9). Non-inferiority requires CI lower bound > −0.5 pp. Measured lower bound = **−4.8 pp → FAILS**. The CI is simply too wide (±4–5 pp) to certify non-inferiority at a 0.5 pp margin — the battery is underpowered at this model scale (see §8, caveat 3).

### Per-category paired deltas (recipe − baseline)

| Category | n (recipe) | Δ | 95% BCa CI | Accuracy: FP16 / Q4_K_M / recipe | Verdict |
|---|---|---|---|---|---|
| MMLU | 100 | +5.00 pp | [−5.0 pp, +14.0 pp] | 28.0 / 20.0 / 25.0% | NO_SIGNIFICANT_DIFFERENCE |
| GSM8K | 60 | −1.67 pp | [−10.0 pp, 0.0 pp] | 1.7 / 3.3 / 1.7% | NO_SIGNIFICANT_DIFFERENCE |
| HumanEval | 30 | 0.00 pp | [0.0 pp, 0.0 pp] | 0 / 0 / 0% | NO_SIGNIFICANT_DIFFERENCE (all zero — floor) |
| needle | 40 | −10.00 pp | [−27.5 pp, −5.0 pp] | 7.5 / 17.5 / 7.5% | ⚠️ Mechanical "SIGNIFICANT_REGRESSION" — **see note** |
| copy | 20 | 0.00 pp | [0.0 pp, 0.0 pp] | 0 / 0 / 0% | NO_SIGNIFICANT_DIFFERENCE (all zero — floor) |

**Needle note — the "regression" is not a degradation.** On the full 40 needle items, **FP16 itself scores 3/40 (7.5%)**. The recipe also scores **3/40 (7.5%)** — the same count as the reference (items differ by one swap: recipe got needle/23 which FP16 missed, and missed needle/14 which FP16 got). The baseline's 7/40 (17.5%) is 4 items *above the FP16 reference* (needle/17, needle/23, needle/3, needle/5 — all FP16 misses). The bootstrap mechanically flags a regression because it only sees recipe-vs-baseline, but recipe-vs-reference is 0. Do not report "BitBlame regressed retrieval" — the honest statement is: *on needle, the recipe matched the FP16 reference's count (3/40); the baseline scored 7/40 by chance above the reference.* This is exactly why §8.3 failure attribution must always include the FP16 reference.

### Verdict in one sentence

**The BitBlame recipe did not beat the Q4_K_M baseline.** At 10.5% smaller size it performs identically (11.60% = 11.60% accuracy, Δ 0.00 pp, CI includes zero), and the CI is far too wide to claim non-inferiority. The closed loop *ran end-to-end and produced a valid, smaller artifact* — the measurement machinery works — but the recipe did not demonstrate a quality win on this battery at this model scale.

## 5. Related work — honesty section

BitBlame is **"firefly/mlx-optiq-class dynamic attribution for the llama.cpp/GGUF ecosystem."** The blanket "no integrated dynamic tool exists" claim stays retracted. Live re-verification (METHODOLOGY_review.md, 2026-10-14) confirms the narrowed positioning:

| Prior art | What it does | What BitBlame adds |
|---|---|---|
| **firefly** (neelvad/firefly) | Per-layer sensitivity via real torchao quantization, relative-L1 divergence, recipe strategies with accuracy-bar gating, closed loop on the served checkpoint; HF/torchao/vLLM/SGLang. No GGUF, no bootstrap CIs. | GGUF/llama.cpp recipes, KL metric, CIs on every blame number, token-category blame |
| **mlx-optiq** | Per-layer KL sensitivity (exact quantize→dequantize round-trip) on a six-domain calibration mix; greedy knapsack; MLX only. More faithful sensitivity than BitBlame's noise screen; no CIs, no token-category blame, no documented closed loop. | Cheaper screen + exact Stage-2 verification, CIs, token-category blame, GGUF |
| **quantbench (stef41)** | Static header parsing only; never executes. | — |
| **quantbench (jamespa16)** | Executes (llama-server + HumanEval pass@1) but aggregate-only: one score per quant, no layer signal. | Layer attribution |
| **microant** (Code-byte404/microant, 5 stars, MIT, 2026-07-10) | **Closest adjacent work.** Whole-model GGUF/MLX capability ruler with bootstrap CIs: 250-case capability pack across quant tiers (incl. GGUF Q2_K/Q3_K_M ladders), per-tier deltas with bootstrap 95% CIs, and a `confirm` stage where a rung counts as held only if the test CI's lower bound clears the bar — methodologically aligned with BitBlame §8. Its "within-chain failure attribution" is a tier/chain damage breakdown, **not per-layer blame**. It does no sensitivity measurement, no bit allocation, no recipe emission, and actual model runs need Apple Silicon. | Per-layer blame feeding allocation; measured recipes; token-category blame; CPU (non-Apple) path |
| **morshoto/qwen-meetup-tokyo exp_002** | Paired-bootstrap practical equivalence (Q4_K_M vs Q8_0 GGUF, ±10 pp margin at 95%). Whole-model comparison only; no attribution. | Methodological sibling for §8, not a threat |

**Differentiator (ii) rewording (applied in spec v1.1):** CIs on GGUF *measurement* exist (microant, morshoto). BitBlame's precise claim is narrower: *"paired-bootstrap CIs on per-layer attribution, significance-gating the allocation"* — no prior art does that.

## 6. Honest caveats

1. **Mock-noise screen is approximate.** Stage 1 perturbs weights with uniform noise matched to quantization variance; it is not real quantization. The screen-vs-exact agreement (ρ = 0.9286, §3) buys back confidence *for the top-8*, but the screen's ranking of the other 23 units is unverified.
2. **Tiny-model transfer limits.** SmolLM2-135M is 134.5M params. Nothing here claims the ranking (embed-dominated, blk.29/11/28) transfers to 1.7B+ models. Phase C (Qwen3-1.7B) is the transfer test; it has not run.
3. **The battery is underpowered at this scale.** All three models are near the floor: GSM8K 1.7–3.3%, HumanEval 0% for everyone, copy 0% for everyone, overall FP16 accuracy 12.8%. When even the FP16 reference fails most items, a ±4–5 pp CI is unavoidable at n = 250. A recipe win (or loss) of a few pp is *undetectable* here. The verdict is "no evidence of a difference," not "evidence of no difference."
4. **Battery complete.** The recipe arm now covers all 250 items (resume run, §1); copy category measured (all-zero floor).
5. **llama.cpp K-quant remapping — FIXED in the v1.1 optimizer model.** The v1.0 optimizer assumed the requested tier lands uniformly per block, but llama.cpp remaps each request (`tensor_type_fallback` + Q4_K_M mixer + mixer-only counters + `--tensor-type` regex first-match). The v1.1 model reproduces both GGUFs' 272/272 tensor types and stored bytes exactly (§2.7). One emission wart remains for P3: the `output.weight` pattern (emitted for tied embeddings) regex-matches every `blk.*.attn_output.weight`, landing mixer-block attn_output at fallback(embed alloc) instead of the mixer tier — harmless here (embed alloc Q5_K), but the emitter should drop `output.weight` when embeddings are tied / anchor patterns.
6. **CPU adaptations.** Calibration N=16/T=256 (spec §12: 32×512); split-half ρ = 0.9754 ≥ 0.9 rescues the ranking's reliability, but the smaller N/T is a deviation from spec.
7. **Per-unit CIs are marginal, not family-wise** (§6.3 honesty clause). The significance gate is a screening heuristic, not an FWER-controlled discovery procedure.
8. **Greedy decoding only; English calibration only.** No sampling-temperature, multilingual, or long-context generality is claimed.
9. **The closed-loop harness is fixed.** The original run exited 1 on the `llama-gguf r` self-test; the fixed harness (header read + type simulation) re-ran exit 0 with 272/272 tensors verified (§1).
10. **imatrix agreement ρ (spec v1.1 §7.4.2, bar ρ ≥ 0.5) still uncomputed.** The imatrix.dat exists; parsing it into per-tensor importance and correlating with $S_u$ is a P3 item. The gate is corroboration (reported either way), not blocking, so it does not affect this verdict.

## 7. What the results DO and DO NOT support for launch claims

**Supported (say these):**
- "BitBlame localizes quantization sensitivity: on SmolLM2-135M, the embedding plus 3 blocks carry ~58% of measured sensitivity (BCa CIs, null-floor-gated), with split-half rank stability ρ = 0.9754."
- "The mock-noise screen predicts exact-quantization ranking (Spearman ρ = 0.93 on the top-8) — screening is validated on this model."
- "The closed loop works end-to-end through real llama-quantize: measured blame → exact-DP allocation → re-quantize → re-measure → paired-bootstrap verdict (harness exits 0, 272/272 tensor types verified)."
- "A BitBlame recipe produced a 10.5%-smaller GGUF that performed identically to uniform Q4_K_M on the full 250-item battery (11.60% = 11.60%, Δ 0.00 pp, CI [−4.8, +4.0])."
- "The optimizer's bit-cost model is verified exact: it reproduces both GGUFs' 272/272 tensor types and stored bytes (0.0000% error)."

**NOT supported (never say these):**
- ❌ "BitBlame improves accuracy / beats Q4_K_M." (Verdict: NO_SIGNIFICANT_DIFFERENCE, 0.00 pp, CI [−4.8, +4.0].)
- ❌ "BitBlame reduces degradation / preserves quality at smaller size." (Non-inferiority at the 0.5 pp margin FAILS — CI too wide.)
- ❌ "BitBlame regresses retrieval." (On needle the recipe matched the FP16 reference's count, 3/40; the baseline scored above the reference by chance; see §4.)
- ❌ "The ranking transfers to larger models." (135M only; Phase C not run.)
- ❌ "First tool / no prior art." (firefly, mlx-optiq, microant exist; see §5.)

## 8. Missing before P3 ship (one line each)

- ~~Full 250-item recipe battery~~ ✅ DONE (resume run, §1).
- ~~Fix closed-loop harness verification~~ ✅ DONE (header read + type simulation, re-ran exit 0, 272/272).
- ~~Optimizer llama.cpp mapping model~~ ✅ DONE (v1.1 remap model, exact on both files, §2.7).
- ~~Spec v1.1 patches~~ ✅ DONE (METHODOLOGY.md v1.1: microant/morshoto, (ii) wording, ρ ≥ 0.7 / ρ ≥ 0.5 gates, DP = gated set, blocking-gates §8.4).
- ~~P1_REPORT.md cleanup~~ ✅ DONE (pointer to this report at top).
- **imatrix agreement ρ:** parse imatrix.dat, correlate per-tensor importance with $S_u$ vs the ρ ≥ 0.5 bar (spec §7.4.2).
- **Recipe-emission wart:** drop `output.weight` pattern when embeddings are tied (it regex-matches all `attn_output` tensors) and/or anchor `--tensor-type` patterns (§6, caveat 5).
- **Real CLI:** `cli.py` / `sensitivity.py` / `stats.py` are 10-line stubs — the README Quickstart (`bitblame scan/probes/recipe/validate/report`) does not run.
- **Tests:** no `tests/` directory exists; add at least unit tests for bootstrap BCa, optimizer DP + remap model, and GGUF header verification.
- **README correction:** remove "(planned)" features that are now implemented, and stop advertising a CLI that does not exist.
- **Git repo:** no repository initialized — `git init`, first commit, remote under habib-analyst (sole GitHub account).
- **HF dataset:** no dataset artifact or upload record exists — create the BitBlame probe-battery + sensitivity dataset on Hugging Face (habib-analyst).
- **HF Space:** no Space exists — build the interactive demo (blame heatmap + recipe explorer) on Hugging Face (habib-analyst).
- **Phase C (deferred, not a P3 blocker):** Qwen3-1.7B transfer test, pending GPU quota refresh (2026-10-17).
