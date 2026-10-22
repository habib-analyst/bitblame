"""Phase B — Stages 0/1: LOLO mock-noise scan on SmolLM2-135M (CPU).

Writes:
  hidden_files/phaseB_scan.npz       — main scan (b=4, R=2)
  hidden_files/phaseB_null.npz       — null-floor scan (b=8, R=1)
  hidden_files/phaseB_calib.json     — calibration sequence metadata
  hidden_files/phaseB_progress.jsonl — per-batch progress log

Calibration corpus: FineWeb-Edu sample (§5.1), tokenized with the model's
own tokenizer; sequences with >5% non-ASCII or code fences dropped.
N=16 x T=256 (CPU-time adaptation of §12's 32x512; the split-half gate in
§2.3 decides sufficiency — documented in P1_REPORT.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import lolo
from bitblame.config import ScanConfig
from bitblame.modeling import list_units, load_hf_model

ROOT = "/home/hatch/workspace/bitblame"
MODEL_DIR = f"{ROOT}/data/models/SmolLM2-135M"
HID = f"{ROOT}/hidden_files"


def build_calibration(tokenizer, n_seqs: int, seq_len: int, seed: int,
                      batch_size: int = 2):
    """FineWeb-Edu sample -> N token sequences of exactly seq_len tokens.

    Source: HuggingFaceFW/fineweb-edu, sample/10BT/000_00000.parquet, first
    row group fetched via HTTP range requests (the full 2 GB+ file is not
    downloaded); 60 documents cached locally.
    """
    df = pd.read_parquet(f"{ROOT}/data/fineweb-edu-docs.parquet")
    text_col = "text" if "text" in df.columns else df.columns[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    seqs, meta = [], []
    for idx in order:
        if len(seqs) >= n_seqs:
            break
        text = str(df.iloc[idx][text_col])
        # §5.1 preprocessing: drop >5% non-ASCII or code fences
        if "```" in text:
            continue
        ascii_frac = sum(ord(c) < 128 for c in text) / max(1, len(text))
        if ascii_frac < 0.95:
            continue
        ids = tokenizer(text, truncation=False)["input_ids"]
        if len(ids) < seq_len:
            continue
        seqs.append(ids[:seq_len])
        meta.append({"doc_row": int(idx), "n_tokens_doc": len(ids),
                     "ascii_frac": round(ascii_frac, 4)})
    assert len(seqs) == n_seqs, f"only {len(seqs)} usable docs (need {n_seqs})"
    with open(f"{HID}/phaseB_calib.json", "w") as f:
        json.dump({"n_seqs": n_seqs, "seq_len": seq_len, "seed": seed,
                   "batch_size": batch_size,
                   "source": "HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet "
                             "(row group 0 via HTTP range requests; 60 docs cached)",
                   "sequences": meta}, f, indent=2)
    return [torch.tensor(s, dtype=torch.long) for s in seqs]


def save_scan(res: lolo.ScanResult, path: str, extra: dict | None = None):
    np.savez_compressed(
        path,
        unit_ids=np.array(res.unit_ids),
        seq_kl=res.seq_kl, token_kl=res.token_kl, draw_S=res.draw_S,
        outlier_mask=res.outlier_mask,
        tail_S=np.array([t["mean"] for t in res.tail]),
        tail_p50=np.array([t["p50"] for t in res.tail]),
        tail_p95=np.array([t["p95"] for t in res.tail]),
        tail_max=np.array([t["max"] for t in res.tail]),
        bits=np.array(res.bits), n_draws=np.array(res.n_draws),
        runtime_s=np.array(res.runtime_s),
        batch_times=np.array(res.batch_times_s),
    )
    print(f"saved {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="sequences per forward batch (1 = memory-safe)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 units, 4 seqs — plumbing check on the real model")
    args = ap.parse_args()

    import os
    os.makedirs(HID, exist_ok=True)
    torch.set_num_threads(2)
    t0 = time.time()
    prog = open(f"{HID}/phaseB_progress.jsonl", "a")

    def log(**kw):
        kw["elapsed_s"] = round(time.time() - t0, 1)
        prog.write(json.dumps(kw) + "\n")
        prog.flush()
        print(json.dumps(kw), flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    log(event="loading_model")
    model = load_hf_model(MODEL_DIR)
    units = list_units(model, n_layers=30)
    log(event="model_ready",
        units=[u.id for u in units], n_units=len(units))

    if args.smoke:
        units = units[:2]
        args.n_seqs, args.seq_len, args.batch_size = 4, 64, 2

    seqs = build_calibration(tokenizer, args.n_seqs, args.seq_len,
                              seed=20261014, batch_size=args.batch_size)
    batches = [torch.stack(seqs[i:i + args.batch_size])
               for i in range(0, len(seqs), args.batch_size)]
    log(event="calibration_ready", n_batches=len(batches))

    cfg = ScanConfig(n_seqs=args.n_seqs, seq_len=args.seq_len,
                     batch_size=args.batch_size)

    def progress(bi, nb, el):
        log(event="batch", batch=f"{bi}/{nb}",
            eta_min=round(el / bi * (nb - bi) / 60, 1) if bi else None)

    log(event="scan_start", bits=4, n_draws=2)
    scan = lolo.run_lolo_scan(model, units, batches, cfg, bits=4,
                              n_draws=2, salt="phaseB", progress=progress,
                              ckpt_path=f"{HID}/phaseB_scan_ckpt.npz"
                              if not args.smoke else None)
    save_scan(scan, f"{HID}/phaseB_scan.npz")
    S = scan.seq_kl.mean(axis=1)
    order = np.argsort(-S)
    log(event="scan_done", runtime_s=round(scan.runtime_s, 1),
        top5=[(scan.unit_ids[i], round(float(S[i]), 6)) for i in order[:5]])

    log(event="null_start", bits=8, n_draws=1)
    null = lolo.run_lolo_scan(model, units, batches, cfg, bits=8,
                              n_draws=1, salt="phaseB-null", progress=progress,
                              ckpt_path=f"{HID}/phaseB_null_ckpt.npz"
                              if not args.smoke else None)
    save_scan(null, f"{HID}/phaseB_null.npz")
    log(event="null_done", runtime_s=round(null.runtime_s, 1),
        total_s=round(time.time() - t0, 1))
    prog.close()
    print("PHASE B SCAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
