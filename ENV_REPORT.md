# BitBlame — Environment Report (2026-10-14 ~03:25 PKT)

## 1. Kaggle auth status

**Working.** `~/.local/bin/kaggle` (CLI 2.2.4) authenticates via `~/.kaggle/access_token`
(600 perms, new-style token auth — no `kaggle.json`). `kernels list --mine` returned
20 kernels for account **habib734**. All inspection below was strictly read-only:
no kernel was started, stopped, deleted, or re-run; no outputs were downloaded.

## 2. Kernels observed (read-only)

| ref | lastRunTime (UTC) |
|---|---|
| habib734/arc-agi-3-bfs-curiosity-solver | 2026-10-07 03:48:28 |
| habib734/caremanage-triage-family-c | 2026-10-09 15:23:27 |
| habib734/caremanage-careplan-family-b | 2026-10-09 15:22:16 |
| habib734/caremanage-medsafety-family-a | 2026-10-09 15:20:36 |
| habib734/medmnist-28-batch2b | 2026-10-09 23:16:21 |
| habib734/habib-ur-rehman-mabe-xgboost-submission | 2025-12-15 11:23:23 |
| habib734/medmnist-3d-suite-30-runs | 2026-10-06 15:58:51 |
| habib734/rsna-debug | 2026-10-13 22:12:25 |
| habib734/rsna-knee-effnet-efficiency-track | 2026-10-13 22:13:39 |
| habib734/misconception-detection-train-multimodal-v6-staged | 2026-10-12 13:01:53 |
| habib734/medmnist-28-batch3-seed71 | 2026-10-12 11:16:30 |
| habib734/stable-ssl-phase2 | 2026-10-11 17:07:19 |
| habib734/notebookc822dbc0e3 | 2026-10-12 12:58:28 |
| habib734/medmnist-28-batch3 | 2026-10-11 11:15:52 |
| habib734/notebookf5b4734ed1 | 2026-10-12 04:42:13 |
| habib734/medmnist-28-batch2c | 2026-10-11 05:13:36 |
| habib734/medmnist-28-batch2 | 2026-10-09 17:14:52 |
| habib734/medmodel-discovery-run3 | 2026-10-09 11:41:21 |
| habib734/medmodel-discovery-run2 | 2026-10-08 12:09:44 |
| habib734/medmodel-discovery-full-run | 2026-10-08 08:19:08 |

**Status checks (read-only, the two most recent):**
- `habib734/rsna-knee-effnet-efficiency-track` → `KernelWorkerStatus.ERROR`
  (last run 2026-10-13 22:13:39 UTC ≈ 2026-10-14 03:13:39 PKT, minutes before this
  environment task started). **Flag for the bounties-competitions chat** (which owns
  the RSNA submission): the latest RSNA training run errored. Nothing was touched
  or downloaded here.
- `habib734/rsna-debug` → `KernelWorkerStatus.COMPLETE`.

No other kernel statuses were queried (no need; inspection kept minimal).

## 3. GPU quota state (via Kaggle API `quota_view`)

| resource | used | remaining | total | refreshes |
|---|---|---|---|---|
| GPU | **28.48h** | **1.52h** | 30.00h | 2026-10-17T00:00:00 |
| TPU | 0.00h | 20.00h | 20.00h | 2026-10-17T00:00:00 |

**Implication for BitBlame:** only ~1.5 GPU-hours remain this week. All BitBlame
build/validation work must run on local CPU (or wait for the 2026-10-17 refresh).
No Kaggle GPU kernels were launched for this task, per the hard rule.

## 4. Local Python environment

- Python **3.12.3**
- numpy **2.5.4** ✅ (prototype dependency)
- scipy **1.18.1** ✅
- matplotlib **3.6.3** ⚠️ — system copy compiled against NumPy 1.x; import fails
  under numpy 2.5.4 (`numpy.core.multiarray failed to import`). The prototype
  catches this and skips chart generation gracefully (ASCII chart printed instead).
- torch / transformers / datasets / huggingface_hub — **not installed**.

**Decision:** pure-NumPy prototype (no `pip install`). Faster, zero download risk
through the egress proxy, and sufficient to validate the core math. `requirements.txt`
carries the full-stack pins (torch CPU, transformers, datasets, numpy, scipy,
matplotlib) for the real build later.

## 5. Prototype: core measurement loop (key deliverable)

`prototype.py` — pure NumPy, CPU, random init, no downloads.

**Design:**
- Tiny decoder-only transformer: vocab 128, hidden 64, 4 heads, **4 layers**,
  FFN 128, seq len 32, RMSNorm, ReLU FFN. Random weights, fixed seeds.
- **(a)** Per-layer perturbation: uniform noise in `[-Δ/2, +Δ/2]` added to each
  weight matrix (`wqkv`, `wo`, `w1`, `w2`) of one layer at a time, where
  `Δ = 2·max|W| / (2⁸−1)` is the **fake INT8 uniform-quantization step size**
  calibrated per matrix. LayerNorm scales left unperturbed (kept high-precision
  in real quantization).
- **(b)** Forward pass over 8 random calibration token sequences.
- **(c)** KL-divergence `KL(p_base ‖ p_perturbed)` of output distributions,
  averaged over batch × positions.
- **(d)** Layers ranked by KL (higher = more quantization-sensitive).

**Results:**

| layer | KL (seed 7) | rank | KL (seed 1234) | rank |
|---|---|---|---|---|
| L0 | 2.61e-05 | 1 | 1.15e-05 | 1 |
| L1 | 1.13e-05 | 2 | 5.67e-06 | 2 |
| L2 | 7.85e-06 | 3 | 2.57e-06 | 4 |
| L3 | 4.13e-06 | 4 | 3.37e-06 | 3 |

- **Non-uniform ranking confirmed:** ~6.3× spread between most/least sensitive
  layer (seed 7). Earlier layers rank most sensitive — sensible, since their
  perturbations propagate through more downstream computation.
- **Stable where it matters:** top-2 (L0 > L1) identical across both seeds;
  bottom two swap (expected with random weights).
- **Sanity control:** zero-noise perturbation → KL = 0.000e+00 exactly (both seeds).
- **Runtime:** 0.1–0.2 s on CPU (well within "runs in minutes").

Results also saved to `prototype_results.json`.

**Caveat (important):** this validates the *plumbing* only — the perturbation is
synthetic uniform noise (not actual quantized weights) and the model is random
(not trained). It does **not** validate real quantization behavior. The real
build will replace the fake step with genuine INT8/FP8 quantization and run
against real checkpoints on calibration text.

## 6. Repo scaffold

`~/workspace/bitblame/` now contains:
- `README.md` — one-line description + vision (planned features marked as planned)
- `src/bitblame/__init__.py` — package stub, v0.1.0
- `requirements.txt` — torch (CPU index), transformers, datasets, numpy, scipy, matplotlib
- `.gitignore` — standard Python + `*.safetensors`, `*.bin`, `*.gguf`, `data/`, secrets
- `prototype.py` + `prototype_results.json` — the validated core loop
- `ENV_REPORT.md` — this file
