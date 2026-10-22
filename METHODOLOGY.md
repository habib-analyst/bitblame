# BitBlame — Methodology Specification

**Status:** Design spec v1.1 · 2026-10-14 · Author: Muse (for Habib Ur Rehman)
**v1.1 patches (additive; see §15 changelog):** microant + morshoto citations,
differentiator (ii) tightened to per-layer-attribution CIs, screen-vs-exact
agreement gate (ρ ≥ 0.7), imatrix corroboration bar (ρ ≥ 0.5), DP unit set
= gated set clarification, blocking-gates section (§8.4).
**Scope:** Core methodology for dynamic quantization forensics on decoder-only
transformer LLMs. Reference target: Qwen3-1.7B class (28 transformer blocks),
FP16 base vs. llama.cpp k-quant builds, free-tier GPU (Kaggle T4/P100) + CPU.
**Non-goals:** training/fine-tuning, new quantizer invention, production serving.
**Positioning:** firefly/mlx-optiq-class dynamic quantization attribution for the
llama.cpp/GGUF ecosystem, with KL-divergence sensitivity + paired-bootstrap CIs
+ token-category blame. (Honest prior-art accounting in §11 — the blanket
"first integrated tool" claim is false and is not made here.)
**Compute plan:** CPU-first (tiny models, now) → GPU (Qwen3-1.7B, after the
Kaggle GPU refresh on 2026-10-17; only 1.52h remain until then). See §12.

> **How to read the novelty flags.** Every section ends with a flag:
> `[STANDARD]` = established practice (prior art named);
> `[ADAPTED]` = known idea applied to a new problem;
> `[NOVEL]` = to our knowledge, not published/done this way.
> §11 collects the full accounting. If you find prior art for anything marked
> `[NOVEL]`, that flag is wrong — correct it and cite the source.

---

## 0. Notation

- FP16 reference model $M_0$; quantized candidate $M_q$.
- $L$ transformer blocks; **unit** $u$ = one block's weights (default granularity),
  or one (block, tensor-role) pair in fine mode.
- Calibration corpus $C = \{x^{(n)}\}_{n=1}^{N}$, each a token sequence of length $T_n$.
- $p_{n,t}(v) = \mathrm{softmax}(z^{0}_{n,t})[v]$: FP16 next-token distribution at
  position $t$. $q^{(u)}_{n,t}$: same, with unit $u$ mock-quantized.
- Target bit-width $b$; bit-width menu $\mathcal{B} = \{2,3,4,5,6,8\}$
  mapped to llama.cpp types $\{Q2\_K, Q3\_K, Q4\_K, Q5\_K, Q6\_K, Q8\_0\}$.

---

## 1. Pipeline overview

```
 STAGE 0  Cache FP16 logits            (1 forward pass per calibration seq)
   │
 STAGE 1  Mock-noise LOLO scan         (per-unit sensitivity + bootstrap CIs)
   │        └─ null-floor run (8-bit noise ≈ no-op) → measurement floor
   │        └─ split-half reliability gate (ρ ≥ 0.9) or double N
   │
 STAGE 2  Exact top-k verification     (faithful k-quant math in torch,
           + slope calibration)         real block scales, super-blocks)
   │
 STAGE 3  Behavioral probe battery     (FP16 vs baseline-quant, paired items)
   │
 STAGE 4  Bit-budget optimizer         (significance-gated multiple-choice
           knapsack, DP)                → llama.cpp overrides + imatrix
   │
 STAGE 5  Closed-loop validation      (re-quantize, re-measure, pre-registered
                                        paired bootstrap test)
   │
 OUTPUT   blame_report.json · heatmap.png · recipe.sh
```

**Development phases** (GPU quota: 1.52h remain until the 2026-10-17 refresh —
see §12): **Phase A** (CPU, tiny-random transformer): prove the plumbing.
**Phase B** (CPU, SmolLM2-135M/360M): full pipeline end-to-end, including the
real `llama-quantize` closed loop (`llama.cpp` is a CPU toolchain — the loop
does not need GPU). **Phase C** (GPU, post-refresh): Qwen3-1.7B at the budgets
in §5.3. Phase B validates the *machinery*; only Phase C validates 1.7B-scale
*findings*.

**Design principle:** never pay for information twice. Cheap, slightly wrong
estimates (mock noise) screen the space; expensive, exact measurements (real
quantizer math, behavioral probes) are spent only where the cheap screen says
it matters. Every number that enters the optimizer carries a confidence
interval; the optimizer is forbidden from spending bits on statistically
indistinguishable units.

---

## 2. Mock quantization

### 2.1 Definition

For target bit-width $b$, tensor $w$, and quantization group $g$ (chosen to
match the target quantizer's grouping — for llama.cpp k-quants, 256-weight
super-blocks; see §2.4):

$$s_g \;=\; \frac{\max_{i \in g}\,|w_i|}{\,2^{\,b-1} - 1\,}
\qquad\text{(symmetric; per-group scale)}$$

$$\tilde{w}_i \;=\; w_i + \varepsilon_i,
\qquad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{U}(-s_g/2,\; s_g/2)$$

**Validity argument** (why this estimates quantization sensitivity):

1. **Moment matching.** True round-to-nearest quantization error
   $e_i = Q(w_i) - w_i$ satisfies $|e_i| \le s_g/2$ deterministically, and under
   the classical high-resolution assumption (Widrow & Kollár, 2008) is
   approximately $\mathcal{U}(-s_g/2, s_g/2)$ within each bin. Hence
   $\mathbb{E}[\varepsilon_i] = 0 = \mathbb{E}[e_i]$ (first moment, unbiased)
   and $\mathrm{Var}(\varepsilon_i) = s_g^2/12 \approx \mathrm{Var}(e_i)$
   (second moment matched). Sensitivity measured under $\varepsilon$ is
   therefore the sensitivity to noise with the *same support and variance*
   as real quantization error.
2. **Behavioral currency.** The perturbation is scored by its effect on the
   output distribution (§3), not on the weights — so what is estimated is
   *end-to-end behavioral sensitivity*, the quantity the recipe optimizer
   actually needs.
3. **Cheap.** One forward pass per unit; no requantization, no gguf round-trip.

**Known limitations (stated honestly, mitigated by design):**

- (a) Real error is *deterministic given $w$* and correlated with it
  (values near bin centers err less); iid noise misses this. → Mitigated by
  **Stage 2** exact verification with faithful quantizer math.
- (b) k-quants allocate bits non-uniformly *within* a super-block using the
  imatrix; uniform per-group noise ignores importance weighting. → Mitigated
  by calibrating group scales from the real quantizer's block layout and by
  Stage 2, which reproduces the importance weighting.
- (c) The high-resolution assumption degrades for outlier-heavy tensors
  (small sub-blocks dominated by one outlier). → Mitigated by the null-floor
  run (§2.3) and by reporting the deterministic worst-case bound
  $|\varepsilon_i| \le s_g/2$ alongside.

> Flag: `[ADAPTED]`. The uniform quantization-noise model is classical
> (Widrow & Kollár, *Quantization Noise*, 2008). Using quantization noise as a
> *training regularizer* is Fan et al., 2020. Using *variance-calibrated,
> group-matched* noise as an *offline per-unit sensitivity estimator for
> post-training quantization forensics* is, to our knowledge, unpublished.
> Contrast: firefly (v0.6.0) skips the noise model entirely and attributes with
> real torchao kernels — more faithful per measurement, far more expensive per
> unit. BitBlame's mock-noise screen exists precisely to avoid paying
> kernel-level cost for all $L$ units; exactness is bought back in Stage 2
> for the top-$k$ only.

### 2.2 Stage 1 — LOLO (leave-one-layer-out) mock-noise scan

For each unit $u$ (default: one transformer block; all its Linear weights;
norm scales included as their own 1-D units):

1. Copy $M_0$; add fresh mock noise (§2.1, $b_{\text{ref}} = 4$) to **only**
   unit $u$'s weights. Noise is redrawn per batch (seeded:
   `seed = hash(unit_id, draw, batch)`).
2. Forward pass over the $N$ calibration sequences (batched); record per-token
   logits.
3. Compute per-unit sensitivity $S_u$ per §3.
4. Repeat for $R = 2$ independent noise draws; average. The draw-to-draw
   variance is logged as a diagnostic (large variance ⇒ unit dominated by a
   few outlier weights ⇒ flag for Stage 2).

**Cost:** $L \times R$ forward passes over $N$ sequences — for $L=28$, $R=2$,
$N=64$, batch 8: 448 batches ≈ 20–40 min on a T4. This deliberately avoids
$O(L)$ *full requantizations*; a forward pass with additive noise is ~100×
cheaper than a gguf quantize+reload cycle.

**Activation-noise mode (dynamic quantization forensics).** A second,
independent scan adds noise to unit $u$'s *output activations* instead of its
weights, with per-token dynamic scale
$s_{n,t} = \max|a_{n,t}| / (2^{b-1}-1)$ over the activation vector. This models
*dynamic activation quantization* (as distinct from weight quantization) and
is reported as a separate blame column. Default $b=8$ (dynamic INT8 acts).

> Flag: `[ADAPTED]` for LOLO-with-noise (leave-one-out sensitivity is standard
> practice; the noise model + output-KL currency is the adaptation).
> Activation-noise mode: `[NOVEL]` as a built-in second blame axis in a
> forensics tool (dynamic-activation sensitivity is usually eyeballed, not
> measured per unit).

### 2.3 Measurement floor and reliability gates

- **Null-floor run:** repeat the Stage-1 protocol with $b = 8$ mock noise
  (≈ FP16-scale perturbation). The resulting $\{S^{\text{null}}_u\}$ distribution
  is the *estimator noise floor*. A unit passes the significance gate (§7)
  only if its 95% CI lower bound exceeds $\max(0, \text{upper CI of floor})$.
- **Split-half reliability:** split calibration sequences into two halves;
  compute the unit ranking on each; require Spearman $\rho \ge 0.9$. If not,
  double $N$ (up to $N=256$) and re-run. If still failing, the report flags
  "unstable ranking — do not trust fine ordering".

> Flag: `[NOVEL]` — split-half reliability gating and an explicit null-floor
> run are standard psychometric practice but, to our knowledge, not built
> into any quantization sensitivity pipeline.

### 2.4 Stage 2 — exact top-k verification + slope calibration

> **Blocking gate (v1.1, review correction #2).** Stage 2 must report
> Spearman ρ between the Stage-1 $S_u$ ranking and the Stage-2 exact-KL
> ranking for the top-$k$. If **ρ < 0.7**, the mock-noise screen is
> unreliable: grow $k$ (verify more units exactly) or revise the mock
> model — **no allocation may proceed on an unvalidated screen**. The
> gate is blocking: a closed-loop verdict built on a screen that failed
> this gate is not valid. (Phase B measured ρ = 0.9286 at 4-bit — gate
> PASSES; see P2_REPORT.md §3.)

For the top $k = 8$ units by Stage-1 $S_u$ (or all units passing the gate,
whichever is smaller):

1. Reimplement the *actual target quantizer's* dequantized-weight math in
   torch (k-quant: per-super-block scales, sub-block scales, importance
   weighting hooks). **Verify bit-identical dequantization against
   llama.cpp on ≥ 3 sample tensors** before use; the check script and its
   pass/fail are part of the report.
2. For each of the $k$ units, at $b \in \{3, 4, 6\}$: replace the unit's
   weights with their *exactly quantized-then-dequantized* values, forward
   pass, measure output KL (§3). This is exact w.r.t. the deployed artifact
   (up to the torch reimplementation, which was verified in step 1).
3. Fit the per-unit loss-vs-bit-width slope (§7.2) from these 3 points.

**Cost:** $k \times 3$ forward passes ≈ 24 passes ≈ 15–30 min on T4.

> Flag: `[ADAPTED]`. Single-tensor exact requantization for sensitivity is
> done in industry practice; the *two-stage* design (cheap noise screen →
> exact verification of top-k only, with a verified torch reimplementation
> instead of gguf round-trips) is our engineering contribution. No published
> equivalent found.

### 2.5 Scaling path — randomized simultaneous perturbation (optional)

When $L$ is large (70B+, $L=80$) and even LOLO passes are too many: run
$K \ll L$ forward passes; in pass $j$, perturb a random subset of units
(mask $m^{(j)} \in \{0,1\}^L$, $P(m^{(j)}_u = 1) = p = 0.25$, fresh mock noise
per perturbed unit); observe total output KL $Y_j$. Fit

$$Y_j \;=\; \beta_0 + \sum_{u=1}^{L} \beta_u\, m^{(j)}_u \;+\; \text{noise}$$

by OLS/LASSO. **Blame score** $\hat\beta_u$ = expected additional output KL
from perturbing unit $u$, *averaging over random co-perturbation backgrounds*
— i.e., a first-order interaction-aware attribution (Banzhaf-like value).
Diagnostics: require model $R^2 \ge 0.7$; if not, fall back to LOLO on the
top candidates. $K = 96$ recommended for $L \le 80$.

> Flag: `[NOVEL]` in quantization. Inspired by RISE (Petsiuk et al., 2018),
> which uses randomized input masking for *model explanations*; applying
> randomized *layer* perturbation + linear attribution to *quantization
> sensitivity* is, to our knowledge, new. Offered as the scaling path, not
> the default — LOLO is exact and cheaper at 1–8B scale.

---

## 3. Sensitivity metric

### 3.1 Definition

Per-token, directional KL divergence with the **FP16 model as reference**:

$$D_t^{(u)} \;=\; D_{\mathrm{KL}}\!\left(p_{\,\cdot\mid x_{<t}}
\;\middle\|\; q^{(u)}_{\,\cdot\mid x_{<t}}\right)
\;=\; \sum_{v=1}^{V} p_{t}(v)\,
\log\!\frac{p_{t}(v)}{q^{(u)}_{t}(v)}$$

Unit sensitivity (the **blame score**):

$$S_u \;=\; \frac{1}{N}\sum_{n=1}^{N}\;
\frac{1}{T_n}\sum_{t=1}^{T_n} D_t^{(u,n)}$$

Implementation: accumulate in float64; clip $q$ below at $10^{-12}$;
compute log-softmax in float32 then cast. Cache FP16 log-probs once
(Stage 0) and reuse for every unit — the reference never moves.

**Reduction over tokens:** primary = mean (above). Additionally report, per
unit: P50 / P95 / max of $\{D_t^{(u)}\}$ — quantization failures concentrate
on outlier tokens, and the tail tells you what the mean hides. And per token
*category* (§4.2).

### 3.2 Why this metric — justification

1. **It is the expected log-likelihood degradation.** For reference
   distribution $p$,
   $\mathbb{E}_{x \sim p}[-\log q(x)] = H(p) + D_{\mathrm{KL}}(p\|q)$.
   Mean token KL is therefore *exactly* the increase in expected NLL of the
   perturbed model measured against the FP16 model's own predictive
   distribution — no data-distribution noise, no sampling variance from
   argmax. Perplexity deltas against true tokens conflate "model was already
   wrong" with "quantization broke it"; KL against the FP16 reference
   isolates the quantization effect.
2. **Directional, not symmetric.** $D_{\mathrm{KL}}(p\|q)$ with $p$ = FP16
   reference penalizes precisely what matters: probability mass the reference
   assigns that the quantized model loses. The reverse direction would
   penalize the quantized model for being *more* confident — irrelevant.
   This is the standard distillation-loss direction, and the reference here
   is literally the teacher.
3. **Full-distribution, not argmax.** Accuracy drop is binary per item: it
   cannot see calibration collapse, entropy shifts, or near-miss
   redistributions that precede outright failures. KL sees them. (Accuracy
   is still the final validation currency — §8 — where coarseness is
   acceptable because the sample is behavioral and the test is paired.)
4. **Scale-free across layers.** L2 distance on activations is dominated by
   per-layer scale differences (RMSNorm outputs, residual stream magnitudes)
   and has no behavioral meaning. KL on the *output* distribution puts every
   unit on one comparable scale: nats of lost information per token.

### 3.3 Alternatives considered and rejected

| Alternative | Verdict |
|---|---|
| L2 / cosine on hidden activations | Rejected: scale-dominated, no behavioral grounding, incomparable across depths. |
| Perplexity delta on calibration text | Rejected: conflates pre-existing model error with quantization damage; higher variance. |
| Accuracy drop on probes (as *sensitivity* metric) | Rejected for screening: binary, needs huge $n$ for tight CIs, blind to calibration shifts. Reserved for Stage 5 validation. |
| Jensen–Shannon / symmetric KL | Rejected: symmetric divergences dilute the reference; we care about loss *relative to* FP16 behavior. |
| Fisher-weighted perturbation norm | Rejected for v1: Hessian/Fisher trace (HAWQ-style) needs backward passes and measures local curvature, not realized behavioral shift. Forward-only KL is cheaper and directly behavioral. Revisit for 70B scale. |

> Flag: `[STANDARD]` for KL-divergence sensitivity *per se* — mlx-optiq
> already ships a KL-divergence sensitivity pass (Apple MLX only), and KL is
> the standard distillation signal. BitBlame does **not** claim novelty on KL
> itself. Its differentiators on the metric are: (a) KL paired with
> bootstrap CIs on every reported number — novel sub-claim (ii), tightened
> in v1.1 to *"paired-bootstrap CIs on per-layer attribution,
> significance-gating the allocation"* (CIs on GGUF *measurement* exist —
> microant, morshoto — see §11; no prior art puts CIs on per-layer blame
> feeding an allocator); (b)
> token-category KL breakdowns — novel sub-claim (iii); (c) KL-derived blame
> driving GGUF k-quant recipes via measured allocation — novel sub-claims
> (i)/(iv). The NLL-decomposition justification (§3.2.1) is what makes the
> choice principled rather than ad-hoc, not what makes it novel.

---

## 4. Attribution dimensions

Blame is attributed along three orthogonal axes, all computed from the same
Stage-1 forward passes (no extra compute beyond storing per-token KLs).

### 4.1 Units: layers → tensor roles (fine mode)

- **Default:** unit = transformer block (28 units for Qwen3-1.7B).
- **Fine mode:** unit = (block, role),
  role ∈ {`attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_up, mlp_down,
  attn_norm, mlp_norm, embed, lm_head`}.
  28 × ~9 ≈ 250 units — use the randomized attribution path (§2.5) or accept
  ~2–3h scan. Known hot spots to watch (from literature, not our claims):
  `lm_head` and early-block `attn_q`/`mlp_down` are frequently sensitive;
  the report must *measure*, not assume.

### 4.2 Token categories

Every per-token KL is tagged; report mean KL per category with bootstrap CIs:

- **Position bucket:** first 25% / middle 50% / last 25% of sequence —
  diagnoses *error accumulation* (late-token KL ≫ early-token KL ⇒
  autoregressive compounding, the classic quantization failure signature).
- **Outlier tokens:** top 1% by max-abs activation in the FP16 final
  hidden state — tests whether damage concentrates on outlier activations
  (the AWQ/GPTQ motivation, measured rather than assumed).
- **Domain tag** (if calibration corpus is tagged): code / math / prose.

> Flag: `[NOVEL]` as a built-in diagnostic. Error accumulation and outlier
> concentration are discussed anecdotally in quantization literature; no
> open tool reports per-category KL with CIs as a standard forensics
> readout.

### 4.3 What is NOT attributed (v1 limits)

- Individual channels/neurons (README's "planned" item — explicitly out of
  v1 scope; the unit×category matrix is the v1 ceiling).
- Cross-unit interactions beyond the first-order averaging of §2.5
  (the closed-loop validation in §8 measures the true joint effect and
  reports the interaction gap vs. the additive prediction).

---

## 5. Calibration data & behavioral probe battery

Two distinct datasets, two distinct jobs. **Never mix them**: the calibration
corpus tunes the estimator; the probe battery judges the outcome.

### 5.1 Calibration corpus (for Stages 0–2)

- **Source:** FineWeb-Edu, 10B sample, English; deduplicated.
- **Size:** $N = 64$ sequences × 2048 tokens (split-half gate, §2.3, decides
  whether to grow to 128/256).
- **Preprocessing:** tokenize with the model's own tokenizer; drop sequences
  with >5% non-ASCII or code fences (keep the calibration distribution
  "neutral prose" so the probe battery can independently cover code/math).
- **Determinism:** fixed seed; sequence IDs logged in the report.

### 5.2 Behavioral probe battery (Stages 3 & 5) — the failure-mode battery

Designed around *known quantization failure modes*, not generic capability.
Total **250 items**; every item is run on FP16, baseline-quant, and
recipe-quant with **identical prompts, greedy decoding (temperature 0)**,
fixed per-category max-tokens. Paired design: item $i$ is the same item
across variants — this is what gives the bootstrap test its power at small
$n$.

| # | Category | n | Source & sampling | Scoring | Failure mode targeted |
|---|---|---|---|---|---|
| 1 | Factual MCQ | 100 | MMLU test, stratified ~2/subject across all 57 subjects | exact option-letter match | knowledge retrieval brittleness |
| 2 | Multi-step arithmetic | 60 | GSM8K test, stratified by #steps & answer magnitude | exact final numeric match (parse `\boxed{}` / last number) | **error accumulation** over reasoning steps |
| 3 | Code synthesis | 30 | HumanEval, stratified easy/med/hard | pass@1, sandboxed exec, 5 s timeout | syntax precision brittleness |
| 4 | Retrieval fidelity | 40 | **synthetic** needle-in-haystack, contexts 4k/8k/16k, 1–3 needles | exact needle recall | attention-logit precision at long context |
| 5 | Exact copying | 20 | **synthetic** random token-string repetition, lengths 64–512 | exact match | copy fidelity / attention sharpness |

**Why these instead of bigger batteries:** at 1.7B on a T4, greedy generation
runs ~30–50 tok/s. 250 items × ~150 mean tokens ≈ 16 min per model variant;
3 variants ≈ 50 min. The battery fits inside one Kaggle GPU session with the
scan. `[STANDARD]` practice is MMLU/GSM8K/HumanEval subsets with stratification;
`[NOVEL]` is the *curation rationale* (each category maps to a quantization
failure mechanism) plus the synthetic retrieval/copy probes as first-class
quantization diagnostics.

**Scoring protocol (exact):**
- MCQ: prompt ends with "Answer with the letter only."; score 1 iff first
  non-whitespace char == gold letter.
- GSM8K: 8-shot CoT prompt (fixed exemplars, logged); score 1 iff parsed
  final number == gold (tolerance: exact string after normalization).
- HumanEval: standard completion prompt; pass@1 with the reference tests.
- Needle/copy: score 1 iff exact match (needle: all needles present verbatim).

### 5.3 Runtime budget targets (Qwen3-1.7B, single T4 16GB)

| Phase | Work | Budget |
|---|---|---|
| Stage 0 | FP16 logit cache, 64 seqs | ~5 min |
| Stage 1 | 28 units × 2 draws LOLO + null floor | 25–45 min |
| Stage 2 | 8 units × 3 bit-widths exact | 15–30 min |
| Stage 3/5 | probe battery × 3 variants | ~50 min |
| Optimizer + validation stats | CPU | ~5 min |
| **Total** | | **~2–2.5 GPU-hours** |

All CPU-side work (bootstrap, DP, report rendering) runs on the free CPU tier.

---

## 6. Paired-bootstrap confidence intervals

**Role in the design.** Every number that enters the optimizer or the final
verdict carries a confidence interval. The pairing (same sequences / same
probe items across variants) is what gives the intervals power at small $n$:
between-item variance cancels, leaving only the *difference* variance. This
is the signature statistical feature — novel sub-claim **(ii)**, tightened
in v1.1: *"paired-bootstrap CIs on per-layer attribution,
significance-gating the allocation"* — and neither firefly nor mlx-optiq
reports uncertainty at all. (CIs on whole-model GGUF measurement exist:
microant's per-tier bootstrap CIs, morshoto's paired-bootstrap
practical-equivalence test — see §11. They do not put CIs on per-layer
blame feeding an allocator; that is the precise claim.)

### 6.1 Objects of inference and resampling schemes

| # | Estimand | Resampling unit | B | Used for |
|---|---|---|---|---|
| (a) | Per-unit sensitivity $S_u$ | calibration **sequences** (whole sequences — tokens within a sequence are dependent, never resample tokens) | 2,000 | attribution ranking, significance gate |
| (b) | Token-category mean KL | calibration sequences, recompute category means per replicate | 2,000 | §4.2 diagnostics |
| (c) | Mean paired probe gain $\bar\Delta$ | probe **items** | 10,000 | Stage 5 validation test |
| (d) | $P(\text{unit }u \in \text{top-}k)$ | from the replicates of (a): fraction with rank$(u) \le k$ | — | ranking stability readout |

For (a): replicate $b$ draws $N$ sequence indices with replacement from
$\{1..N\}$, recomputes $S_u^{*(b)}$ from the cached per-sequence KLs (no new
forward passes — the bootstrap is pure CPU over cached numbers).

### 6.2 Interval estimator: BCa, 95%

Primary: **BCa (bias-corrected and accelerated)**, Efron 1987. BCa corrects
both median bias and skewness — matters because probe scores are binary
(skewed $\Delta$ distributions) and KL values are heavy-tailed. Fallback:
plain percentile if the acceleration constant $\hat a$ is numerically
unstable ($|\hat a| > 0.2$ or jackknife failure); the report logs which
estimator was used per interval. Level: 95% throughout.

### 6.3 Hypothesis tests

- Validation (c): $H_0: \bar\Delta \le 0$. Bootstrap $p =
  (1 + \#\{\bar\Delta^{*(b)} \le 0\}) / (B+1)$.
- Top-$k$ exact-stage claims: per-unit one-sided bootstrap $p$ for
  $S_u > \text{floor}$, with **Holm correction** across the $k$ units when
  the report claims "significantly sensitive".
- Honesty clause: per-unit CIs in (a) are **marginal**, not family-wise.
  The significance gate (§7.1) is a *screening heuristic*, not an
  FWER-controlled discovery procedure. The report states this verbatim.

### 6.4 How the CIs feed the pipeline

1. **Attribution ranking:** units sorted by $\hat S_u$; heatmap cells show
   95% CI whiskers; column $P(\text{top-}k)$ from (d) quantifies ranking
   stability without pretending the point order is exact.
2. **Significance gate (§7.1):** unit $u$ is eligible for bit-upgrade only if
   $\mathrm{CI}_{\text{low}}(S_u) > \max(0, \mathrm{CI}_{\text{high}}
   (\text{null floor}))$. Bits are never spent on statistically
   indistinguishable units.
3. **Optimizer input:** point estimates $\hat S_u$ for gated units (see §7);
   CI width is reported alongside each allocation as its uncertainty.
   (Extension, not v1: robust allocation optimizing the CI-low loss.)
4. **Validation verdict (§8):** the pre-registered success criterion is
   stated *entirely* in CI terms — no $p$-hacking surface, no post-hoc
   thresholds.

### 6.5 Reproducibility

All resampling uses seeded PRNGs (`numpy.random.default_rng(seed)`); seeds
logged in `blame_report.json`. Re-running with the same seed reproduces
intervals bit-identically.

> Flag: `[STANDARD]` statistics (BCa bootstrap, paired design, Holm) —
> `[NOVEL]` **(ii)** as *wired into* a quantization attribution/allocation
> pipeline: CIs on every blame number, significance-gated allocation,
> $P(\text{top-}k)$ stability readouts, and CI-stated pre-registered
> validation. Neither firefly nor mlx-optiq reports any uncertainty.

---

## 7. Bit-budget optimizer

### 7.1 Formal problem

**Multiple-choice knapsack** over units $u \in \mathcal{U}$ (default: $L$
blocks; fine mode: (block, role) pairs):

- Menu $\mathcal{B} = \{2,3,4,5,6,8\}$ bits, mapped to GGUF types
  $\{Q2\_K, Q3\_K, Q4\_K, Q5\_K, Q6\_K, Q8\_0\}$.
- $\mathrm{size}_u(b) = \mathrm{params}_u \cdot b / 8 \times 1.02$
  (2% scale/metadata overhead; refined per k-quant type in implementation).
- Predicted quality loss $\ell_u(b)$ (§7.2).
- Budget $S_{\max}$: default = measured size of the uniform-$Q4\_K\_M$
  baseline (fixed-size comparison; see §8).

$$\min_{b_u \in \mathcal{B}}\; \sum_{u} \ell_u(b_u)
\quad\text{s.t.}\quad \sum_{u} \mathrm{size}_u(b_u) \le S_{\max}$$

**Significance gate (pre-solve):**
$\mathcal{U}' = \{u : \mathrm{CI}_{\text{low}}(S_u) >
\max(0, \mathrm{CI}_{\text{high}}(\text{floor}))\}$;
$u \notin \mathcal{U}'$ are pinned at $b_{\text{default}} = 4$.
The optimizer may only move bits among units whose sensitivity is real.

**Variant (compression mode):** swap objective and constraint — minimize size
subject to predicted loss $\le$ baseline loss $+$ non-inferiority margin.
Emitted on request; validated per §8 secondary criterion.

### 7.2 Sensitivity-to-loss model and calibration

Measured: $\hat c_u = S_u$ at $b_{\text{ref}} = 4$ (Stage 1; Stage-2 exact
values for top-$k$). Model:

$$\ell_u(b) \;=\; \hat c_u \cdot 2^{-\alpha_u\,(b - b_{\text{ref}})}$$

**Theoretical slope.** Under small-noise linearization, output KL is locally
quadratic in the weight perturbation,
$D_{\mathrm{KL}} \approx \tfrac12 \Delta^\top F \Delta$ ($F$ = Fisher), so
$\mathbb{E}[D_{\mathrm{KL}}] \propto \mathbb{E}\|\Delta\|^2 \propto$
noise variance $\propto s^2 \propto 2^{-2b}$. Hence
$\alpha_{\text{theory}} = 2\ln 2 \approx 1.386$ nats-per-bit decay.
This is a *derivation*, not a fit — it tells the optimizer the right
functional form.

**Empirical calibration.** For Stage-2 units, exact KL is measured at
$b \in \{3,4,6\}$; fit $\log \ell_u(b) = \log \hat c_u - \alpha_u(b -
b_{\text{ref}})$ by least squares, then shrink toward theory:

$$\alpha_u \;=\; \lambda\,\hat\alpha_u + (1-\lambda)\,\alpha_{\text{theory}},
\qquad \lambda = R^2_{\text{fit}}$$

($R^2 < 0.5 \Rightarrow \lambda = 0$: fall back to theory; the 2-bit regime
breaks linearization, so never extrapolate the fit below $b=3$ — clamp.)
Units without Stage-2 data use $\alpha_{\text{theory}}$.

**Additivity caveat (stated):** $\sum_u \ell_u(b_u)$ assumes independent
unit contributions. The closed loop (§8) measures the true joint effect and
reports the **interaction gap** (predicted $\sum\ell$ vs. measured KL
reduction); a large gap is itself a forensics finding (super/sub-additive
quantization damage).

### 7.3 Solver: exact DP

> **Clarification (v1.1, review correction #4).** The DP's unit set IS
> the gated set $\mathcal{U}'$ from §7.1 — the DP optimizes only over
> units that passed the significance gate; non-gated units are pinned at
> $b_{\text{default}} = 4$ before the DP runs and never enter it.

$|\mathcal{U}| \le \sim$250, $|\mathcal{B}| = 6$, budget discretized to MB
($S_{\max} \sim$ 1000–1200 MB for 1.7B). Multiple-choice knapsack DP:

```
dp[s]    = min predicted loss using processed units at total size s (MB)
choice[u][s] = bit-width achieving it
initialize dp[0]=0, dp[>0]=inf
for each unit u:
    newdp = inf
    for s in 0..Smax:
        for b in B:
            s2 = s + round(size_u(b))
            if s2 <= Smax: newdp[s2] = min(newdp[s2], dp[s] + loss_u(b))
    dp = newdp   # (keep argmin for traceback)
trace back choices → {b*_u}
```

Complexity $O(|\mathcal{U}|\cdot|\mathcal{B}|\cdot S_{\max}) \approx
250 \times 6 \times 1200 \approx 1.8\text{M}$ ops — milliseconds on CPU.
Fallback for very large $\mathcal{U}$: greedy by marginal gain per byte
($\Delta\ell / \Delta$size); the report logs which solver ran.

> Flag: exact DP for the knapsack is `[STANDARD]` algorithms. **Novel (iv)**:
> *true K-bit-per-layer knapsack in the GGUF world* — firefly allocates
> binary (keep-FP vs quantize); mlx-optiq allocates by greedy knapsack;
> HAWQ-V3 used ILP (needs a solver dependency). BitBlame is exact,
> dependency-free DP over $K=6$ real GGUF quant types, significance-gated.

### 7.4 Recipe emission — driving llama.cpp (novel sub-claim i)

The optimizer output is compiled to a copy-paste `recipe.sh`:

1. **Per-tensor overrides** via llama.cpp's `--tensor-type` mechanism
   (GGUF tensor names, e.g. `blk.0.attn_q.weight`,
   `blk.17.ffn_down.weight`; regex groups allowed):
   ```sh
   llama-quantize --imatrix imatrix.dat \
     --tensor-type blk.0.attn_q.weight=Q6_K \
     --tensor-type 'blk\.(1[6-9]|2[0-7])\.ffn_down\.weight=Q3_K' \
     --tensor-type output.weight=Q6_K \
     model-f16.gguf model-bitblame.gguf Q4_K_M
   ```
   Unmentioned tensors fall back to the base type (`Q4_K_M`). Exact
   flag/regex syntax is pinned against the llama.cpp revision recorded in
   the build manifest — verify before running.
2. **imatrix guidance:** the recipe ships (a) the calibration mix used
   (`imatrix.dat` generated by `llama-imatrix` on the §5.1 corpus), and
   (b) an **agreement diagnostic**: Spearman rank correlation between the
   imatrix per-tensor importance and BitBlame's $S_u$. **Corroboration
   bar (v1.1, review correction #3): pre-registered Spearman ρ ≥ 0.5.**
   Agreement ⇒ independent corroboration; disagreement ⇒ a named tensor
   where the two methods diverge, flagged for inspection. The number is
   reported either way; a below-bar value does not invalidate the
   allocation but must be disclosed with the diverging tensors named.
   (Phase B: not yet computed — P3 item; see P2_REPORT.md §8.)
3. **Why this matters:** llama.cpp's built-in quantizer allocates bits by
   pure heuristics (verified in `src/llama-quant.cpp` — no measurement
   drives its per-tensor choices). BitBlame replaces heuristic allocation
   with *measured-sensitivity-driven* allocation for the GGUF ecosystem.
   No existing tool does this — firefly targets torchao/vLLM, mlx-optiq
   targets MLX.

> Flag: `[NOVEL]` **(i)**. The five-way differentiation (i)–(v) is
> consolidated in §11.

---

## 8. Closed-loop validation

**What it is:** re-quantize with the emitted recipe using the real
`llama-quantize`, re-run the full probe battery, and apply a
**pre-registered** paired statistical test at fixed model size.
Novel sub-claim **(v)** — firefly re-evaluates inside torchao/vLLM;
BitBlame closes the loop through the actual llama.cpp toolchain, on CPU.

### 8.1 Protocol

1. Build three artifacts from the same FP16 source:
   - $M_0$: FP16 reference.
   - $M_{\text{base}}$: uniform `Q4_K_M` (the baseline the recipe must beat).
   - $M_{\text{recipe}}$: `recipe.sh` output (§7.4).
2. Run the §5.2 probe battery (250 items) on all three: identical prompts,
   greedy decoding, fixed seeds. Score per §5.2.
3. Paired differences per item: $\Delta_i =
   \mathrm{score}_{\text{recipe}}(i) - \mathrm{score}_{\text{base}}(i)$,
   $i = 1..250$.

### 8.2 Pre-registered success criteria (fixed before running)

- **Primary (fixed size):** 95% BCa CI on $\bar\Delta$ ($B=10{,}000$) has
  **lower bound $> 0$**, AND $\mathrm{size}(M_{\text{recipe}}) \le
  1.01 \times \mathrm{size}(M_{\text{base}})$.
  Verdict vocabulary: `SIGNIFICANT_GAIN` / `NO_SIGNIFICANT_DIFFERENCE` /
  `SIGNIFICANT_REGRESSION`.
- **Secondary (compression):** with the §7.1 compression variant,
  non-inferiority: CI lower bound $> -0.005$ (0.5 pp margin) at
  $\mathrm{size} \le 0.9 \times$ baseline.
- **Secondary endpoints (reported, not gating):** mean-token KL reduction
  on calibration data (recipe vs base); per-category $\bar\Delta$
  (MMLU/GSM8K/HumanEval/retrieval/copy); interaction gap (§7.2).
- Effect size always reported in percentage points alongside the CI —
  significance without magnitude is not a result.

### 8.3 Failure handling

If the primary criterion fails, the report does not just say "no":
it attributes the delta — per-category $\bar\Delta$ with CIs, and
token-category KL (§4.2) on recipe-vs-base — so the forensics loop
continues (e.g. "recipe fixed GSM8K but regressed retrieval: late-token
KL up 2.1× ⇒ attention-logit precision, raise `attn_q` bits in blocks
20–27"). The loop is *diagnostic*, not just a gate.

> Flag: `[STANDARD]` science (pre-registered paired test), `[NOVEL]`
> **(v)** as an automated built-in loop *driving llama-quantize*:
> measure → allocate → re-quantize (real toolchain) → re-measure →
> statistical verdict, with failure attribution that feeds the next
> iteration.

---

### 8.4 Blocking gates (v1.1) — all must pass before a verdict is trusted

| Gate | Requirement | Where |
|---|---|---|
| Split-half rank stability | Spearman ρ ≥ 0.9 (else double $N$, up to 256) | §2.3 |
| Screen-vs-exact agreement | Spearman ρ ≥ 0.7 (else grow $k$ / revise mock model) | §2.4 |
| Significance gate | CI_low($S_u$) > max(0, CI_high(null floor)) to move bits | §7.1 |
| Size gate (primary) | size(recipe) ≤ 1.01 × size(base) | §8.2 |
| imatrix corroboration | Spearman ρ ≥ 0.5; below-bar ⇒ disclose + name diverging tensors (not blocking) | §7.4.2 |

A closed-loop verdict built on a failed blocking gate is not valid; the
report must say which gate failed and what was done about it.

## 9. Output artifacts

- **`blame_report.json`** — machine-readable: per-unit
  `{S_u, ci_low, ci_high, p_topk, noise_draw_variance, gate_pass}`,
  null-floor stats, split-half $\rho$, token-category KL table with CIs,
  Stage-2 exact values + fitted $\alpha_u$, optimizer allocation
  `{unit: quant_type}`, predicted vs measured loss, validation
  `{delta_bar, ci, p_value, verdict}`, seeds, llama.cpp revision, corpus IDs.
- **`heatmap.png`** — layers (y) × tensor roles (x), cell color = $\hat S_u$,
  whisker/CI annotation, gated-out cells dimmed; second panel: token-category
  KL bars with CIs; third panel (post-validation): per-category $\bar\Delta$.
- **`recipe.sh`** — copy-paste §7.4 commands (imatrix gen + quantize).
- All artifacts versioned with the spec version and config hash.

---

---

## 10. Compute budget

| Phase | Model | Work | Where | Budget |
|---|---|---|---|---|
| A — plumbing | tiny-random transformer (~10M, `prototype.py` lineage) | mock noise, KL, LOLO, bootstrap, DP, artifacts | CPU | minutes |
| B — full pipeline | SmolLM2-135M/360M (trained) | Stages 0–5 end-to-end incl. real `llama-quantize` closed loop | CPU | ~2–4 CPU-hours total |
| C — scale | Qwen3-1.7B | Stages 0–5 per §5.3 | GPU (post-2026-10-17) | ~2–2.5 GPU-hours (T4) |

Phase B is the quota strategy: **the entire methodology — including the
novel closed loop (v) — is proven on CPU before any GPU is spent**, because
`llama-quantize`/`llama-imatrix` are CPU tools and tiny-model forward passes
fit on CPU. GPU is reserved for the one thing CPU cannot do at feasible
speed: 1.7B-scale forward passes. Details in §12.

---

## 11. Novelty & prior art — honest accounting

### 11.1 The blanket claim we do NOT make

BitBlame is **not** the first integrated dynamic quantization-attribution
tool. Four real tools exist (capabilities verified live, 2026-10-14):

- **firefly** (`neelvad/firefly`, v0.6.0): dynamic per-layer attribution
  with **real torchao kernels**, recipe building with verification, and
  closed-loop re-evaluation. Limits: torchao/vLLM backends only; **binary**
  allocation (keep-FP vs quantize); attribution signal is output
  **relative error** (not KL); **no bootstrap CIs**; **no token-category
  attribution**.
- **mlx-optiq** (pip-installable): **KL-divergence sensitivity pass** plus
  **greedy knapsack** bit allocation. Limits: **Apple MLX only**; no CIs;
  no token-category blame; **no documented closed loop**.
- **microant** (`Code-byte404/microant`, 5 stars, MIT, 2026-07-10): **closest
  adjacent work.** Whole-model GGUF/MLX capability ruler with bootstrap
  CIs: 250-case capability pack across quant tiers (incl. GGUF Q2_K/Q3_K_M
  ladders), per-tier deltas with bootstrap 95% CIs, and a `confirm` stage
  where a rung counts as held only if the test CI's lower bound clears the
  bar — methodologically aligned with BitBlame §8. Its "within-chain failure
  attribution" is a tier/chain damage breakdown, **not per-layer blame**.
  It does no sensitivity measurement, no bit allocation, no recipe
  emission, and actual model runs need Apple Silicon.
- **morshoto/qwen-meetup-tokyo exp_002**: paired-bootstrap practical
  equivalence (Q4_K_M vs Q8_0 GGUF, ±10 pp margin at 95%). Whole-model
  comparison only; no attribution. Methodological sibling for §8, not a
  threat.

**Honest positioning (the only claim this spec makes):**

> BitBlame is firefly/mlx-optiq-class dynamic quantization attribution
> **for the llama.cpp/GGUF ecosystem**, with KL-divergence sensitivity +
> **paired-bootstrap CIs** + **token-category blame**.

### 11.2 Differentiation matrix

| Capability | firefly v0.6.0 | mlx-optiq | BitBlame (this spec) |
|---|---|---|---|
| Ecosystem | torchao / vLLM | Apple MLX | **llama.cpp / GGUF** |
| Attribution signal | output relative error | KL divergence | KL divergence (not claimed novel) |
| Perturbation | real torchao kernels | KL sensitivity pass | variance-calibrated mock-noise screen → exact k-quant verification (§2) |
| Uncertainty quantification | none | none | **paired-bootstrap BCa CIs on per-layer attribution; significance-gated allocation; $P(\text{top-}k)$** — (ii) (v1.1 wording: CIs on whole-model GGUF *measurement* exist — microant, morshoto — the novel claim is CIs on *per-layer blame feeding the allocator*) |
| Token-category blame | no | no | **position-bucket / outlier-token / domain KL with CIs** — (iii) |
| Bit allocation | binary (FP vs quant) | greedy knapsack | **exact DP multiple-choice knapsack over $K{=}6$ GGUF types** — (iv) |
| Recipe output | torchao/vLLM recipe | — | **`--tensor-type` overrides + imatrix guidance; measured allocation replacing llama.cpp heuristics** — (i) |
| Closed loop | re-evaluation (same backend) | not documented | **re-quantize via real `llama-quantize` + re-measure + pre-registered paired test** — (v) |

### 11.3 The five genuinely novel sub-claims

1. **(i) Measured-sensitivity-driven GGUF recipes.** llama.cpp's own
   quantizer allocates bits by pure heuristics (verified in
   `src/llama-quant.cpp`); no tool drives its per-tensor `--tensor-type`
   allocation from measured sensitivity, with imatrix-agreement diagnostics.
2. **(ii) Paired-bootstrap CIs on per-layer attribution,
   significance-gating the allocation.** Neither tool reports
   uncertainty; BitBlame puts BCa intervals on blame scores, gates
   allocation on significance, and states validation verdicts in CI terms.
   (v1.1 tightening: CIs on GGUF *measurement* exist — microant's per-tier
   bootstrap CIs, morshoto's paired-bootstrap equivalence test — so the
   claim is precisely CIs on *per-layer blame feeding the optimizer*,
   which no prior art does.)
3. **(iii) Token-category attribution.** Position-bucket / outlier-token /
   domain KL breakdowns as a built-in forensics readout — absent from both.
4. **(iv) True K-bit-per-layer knapsack in GGUF world.** Binary (firefly)
   and greedy (mlx-optiq, HAWQ-V3 used ILP) → exact dependency-free DP over
   six real GGUF quant types.
5. **(v) Closed-loop re-quant + re-measure driving `llama-quantize`.**
   The loop runs through the actual GGUF toolchain with a pre-registered
   statistical verdict and failure attribution that feeds the next
   iteration.

### 11.4 What is explicitly NOT novel

Per-layer sensitivity analysis for mixed precision (HAWQ/HAWQ-V2/HAWQ-V3 —
V3 formulated allocation as ILP); KL divergence as a comparison signal
(mlx-optiq, distillation literature); knapsack-style bit allocation
(HAWQ-V3, mlx-optiq); perturbation-based screening (adapted from Fan et
al., 2020); closed-loop evaluation as a concept (firefly); bootstrap CIs
as statistics (Efron). The mock-noise estimator's validity rests on the
classical Widrow quantization-noise model.

### 11.5 Correction policy

Novelty flags are falsifiable. If prior art surfaces for any of (i)–(v),
the flag is downgraded and the source cited — the differentiation matrix
above is the contract.

> Overall-pipeline flag: `[NOVEL]` **only** in the narrow sense of §11.1–11.3:
> the first open tool combining dynamic attribution **with** bootstrap-CI
> rigor **and** token-category blame **for** the GGUF ecosystem, closing the
> loop through `llama-quantize`. Nothing broader is claimed.

---

## 12. CPU-first development path (GPU quota: 1.52h until 2026-10-17)

**Key insight:** the only GPU-bound operations are neural forward passes at
1.7B scale. Everything else — the mock-noise math, KL/CI/DP machinery, and
*the entire closed loop* (`llama-imatrix` + `llama-quantize` are CPU tools)
— runs on CPU. So the methodology is developed and validated in three
phases, spending GPU only where CPU is infeasible.

### Phase A — plumbing on a tiny-random transformer (CPU, now)

Extend the existing `prototype.py` lineage (pure NumPy/torch, ~10M random
params). Prove: mock-noise injection per §2.1, per-token KL per §3,
LOLO loop, BCa bootstrap code (§6), DP solver (§7.3), JSON/heatmap/
`recipe.sh` emission. **Proves the code runs; proves nothing about real
models.** Time: minutes.

### Phase B — full pipeline on SmolLM2-135M/360M (CPU, now)

Run Stages 0–5 end-to-end on a *trained* tiny model, on CPU:

- **Scan cost (135M):** 30 layers × 2 draws × 32 seqs × 512 tokens,
  batch 8 → 240 batches ≈ 20–40 CPU-min. (Reduce $N$ to 32; the split-half
  gate in §2.3 decides if that's enough.)
- **Probes on CPU:** 135M generates ~20–50 tok/s on CPU; 250 items ×
  ~100 mean tokens ≈ 10–20 min per variant; 3 variants ≈ 30–60 min.
- **Closed loop on CPU:** `llama-imatrix` + `llama-quantize` (real
  toolchain) on a 135M GGUF — minutes. This validates novel sub-claims
  **(i)** and **(v)** with zero GPU.
- **What Phase B proves:** the machinery works on a real trained model;
  CI machinery behaves (coverage sane, gates fire); the optimizer emits
  valid recipes; the validation test has power to detect a real
  recipe-vs-baseline difference; `recipe.sh` actually runs.
- **What Phase B does NOT prove:** that 135M sensitivity rankings transfer
  to 1.7B (outlier structure and depth differ — treat transfer as an open
  empirical question for Phase C, not an assumption).

### Phase C — scale to Qwen3-1.7B (GPU, after 2026-10-17 refresh)

Full run per §5.3 (~2–2.5 GPU-hours on a T4), reusing Phase-B-validated
code unchanged. Validates: ranking stability at depth, the slope
calibration (§7.2) on a real target, and the primary success criterion
(§8.2) at the scale users care about.

### What strictly needs GPU (and what does not)

| Needs GPU | Does NOT need GPU |
|---|---|
| 1.7B forward passes at feasible speed (CPU ≈ 20–40h for the scan alone) | All of Phase A and Phase B |
| Any claim about Qwen3-1.7B specifically | Mock-noise math, KL, bootstrap, DP optimizer |
| | `llama-imatrix` / `llama-quantize` / recipe emission (CPU toolchain) |
| | Closed-loop validation on tiny models (proves the loop, not the scale) |

**Rule until 2026-10-17:** no GPU work. If Phase B surfaces a methodology
bug, fix it on CPU — GPU hours are spent only on a frozen, Phase-B-green
pipeline.

---

## 13. Limitations & non-goals (v1)

1. **GGUF/llama.cpp only.** torchao/vLLM/MLX ecosystems are firefly's and
   mlx-optiq's territory; BitBlame does not compete there in v1.
2. **Additive loss assumption** (§7.2): interactions are measured, not
   modeled — the interaction gap is reported, not optimized.
3. **Mock-noise fidelity**: §2.1 limitations (a)–(c); Stage 2 exists
   because the screen is approximate.
4. **Greedy decoding only** for probes; sampling-based behavioral shifts
   are out of scope.
5. **English calibration corpus**; multilingual quantization behavior not
   covered in v1.
6. **No channel/neuron attribution** in v1 (unit × category is the ceiling).
7. **Tiny-model transfer** (Phase B → Phase C) is unvalidated until Phase C
   runs — no 1.7B claims before then.
8. Non-goals restated: no training, no new quantizer, no serving, no paper.

---

## 14. References

- Yao et al., *HAWQ: Hessian AWare Quantization of Neural Networks with
  Mixed-Precision* (2019); *HAWQ-V2* (2020); *HAWQ-V3: Dyadic Neural Network
  Quantization* (2020) — Hessian sensitivity + ILP allocation.
- Fan et al., *Training with Quantization Noise for Extreme Model
  Compression* (2020) — quantization noise as training regularizer.
- Widrow & Kollár, *Quantization Noise* (Cambridge UP, 2008) — uniform
  noise model, high-resolution assumption.
- Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of
  Black-box Models* (2018) — inspiration for §2.5.
- Efron, *Better Bootstrap Confidence Intervals* (1987) — BCa.
- Frantar et al., *GPTQ* (2022); Lin et al., *AWQ* (2023) — context on
  outlier-aware quantization and sensitive projections.
- firefly (`neelvad/firefly`, v0.6.0) — torchao/vLLM dynamic attribution,
  binary allocation, closed-loop re-evaluation.
- microant (`Code-byte404/microant`, MIT, 2026-07-10) — whole-model
  GGUF/MLX capability ruler: 250-case capability pack across quant tiers,
  per-tier deltas with bootstrap 95% CIs, `confirm` stage with CI-lower-bound
  gating. Closest adjacent work; no per-layer blame, no allocation, no recipes.
- morshoto/qwen-meetup-tokyo exp_002 — paired-bootstrap practical
  equivalence (Q4_K_M vs Q8_0 GGUF, ±10 pp margin at 95%). Whole-model only.
- mlx-optiq — MLX KL-sensitivity + greedy knapsack allocation.
- llama.cpp (`src/llama-quant.cpp`, `llama-imatrix`, `llama-quantize`) —
  GGUF toolchain; k-quant super-block structure.
- Hendrycks et al., *MMLU* (2020); Cobbe et al., *GSM8K* (2021);
  Chen et al., *HumanEval* (2021); *FineWeb-Edu* (2024).

---

*End of spec v1.1. Implementation order: Phase A → Phase B (CPU, now) →
Phase C (GPU, post-2026-10-17). Methodology changes require a spec version
bump and an entry in the repo CHANGELOG.*

---

## 15. Changelog (spec versions)

- **v1.0 (2026-10-14):** initial locked design (§§0–14).
- **v1.1 (2026-10-14, additive patches per METHODOLOGY_review.md
  correction policy §11.5):** (1) microant + morshoto cited as
  measurement-layer prior art (§11.1, §11.2, §14); differentiator (ii)
  tightened to *"paired-bootstrap CIs on per-layer attribution,
  significance-gating the allocation"* (§3.3, §6, §11.2, §11.3).
  (2) Screen-vs-exact agreement gate: Spearman ρ ≥ 0.7, blocking (§2.4,
  §8.4). (3) imatrix corroboration bar: pre-registered Spearman ρ ≥ 0.5,
  reported either way (§7.4.2, §8.4). (4) DP unit set = gated set 𝒰′
  clarification (§7.3). (5) Blocking-gates table (§8.4). (6) Optimizer
  bit-cost model made llama.cpp-remap-aware: K-quant fallback demotion
  (Q2_K/Q3_K→Q4_0, Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0 on ncols % 256 ≠ 0)
  + Q4_K_M mixer + mixer-only counters + `--tensor-type` regex
  first-match semantics, verified to reproduce both Phase-B GGUFs'
  272/272 tensor types and stored bytes exactly (§7.1/§7.4 note;
  `src/bitblame/optimizer.py`, `scripts/verify_recipe_sizes.py`).
