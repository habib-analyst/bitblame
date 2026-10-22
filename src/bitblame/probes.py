"""Behavioral probe battery — §5.2 of METHODOLOGY.md (spec v1.0).

Designed around known quantization failure modes (not generic capability).
250 items total; every item runs on each model variant with identical
prompts, greedy decoding (temperature 0), fixed per-category max-tokens.
Paired design: item i is the same item across variants.

  # | category              |  n | source & sampling
  1 | factual MCQ           |100 | MMLU test, stratified ~2/subject (57 subjects)
  2 | multi-step arithmetic | 60 | GSM8K test, stratified by #steps & answer magnitude
  3 | code synthesis        | 30 | HumanEval, stratified by solution-length tercile
  4 | retrieval fidelity    | 40 | synthetic needle-in-haystack (1k/2k/4k ctx, 1-3 needles)
  5 | exact copying         | 20 | synthetic random token-string repetition (64-512)

Scoring: MCQ exact letter; GSM8K exact final numeric; HumanEval pass@1
(sandboxed exec, 5 s timeout); needle/copy exact match.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from .config import ProbeConfig

LETTERS = "ABCD"


# ---------------------------------------------------------------- builders

def build_mmlu_items(test_dir: str, n: int, seed: int) -> list[dict]:
    """~2 items per subject, stratified across all 57 subjects. Uses 5-shot
    prompting (fixed examples) because the Phase-B model is a base LM, not
    instruction-tuned — zero-shot 'Answer with the letter only' does not
    elicit letter outputs from it (measured 4% < 25% random)."""
    rng = np.random.default_rng(seed)
    subjects = sorted(d for d in os.listdir(test_dir)
                      if os.path.isdir(os.path.join(test_dir, d)) and d != "all")
    assert len(subjects) == 57, f"expected 57 MMLU subjects, got {len(subjects)}"
    # Fixed 5-shot exemplars from diverse subjects (deterministic)
    shots = []
    shot_subjs = ["high_school_mathematics", "high_school_biology",
                  "high_school_us_history", "computer_security", "philosophy"]
    for sj in shot_subjs:
        df0 = pd.read_parquet(os.path.join(test_dir, sj, "test-00000-of-00001.parquet"))
        r0 = df0.iloc[0]
        shots.append(
            f"Question: {r0['question']}\n"
            f"A. {r0['choices'][0]}\nB. {r0['choices'][1]}\n"
            f"C. {r0['choices'][2]}\nD. {r0['choices'][3]}\n"
            f"Answer: {LETTERS[int(r0['answer'])]}")
    fewshot = "\n\n".join(shots) + "\n\n"
    items: list[dict] = []
    per = max(1, round(n / len(subjects)))
    for subj in subjects:
        df = pd.read_parquet(os.path.join(test_dir, subj, "test-00000-of-00001.parquet"))
        idx = rng.choice(len(df), size=min(per, len(df)), replace=False)
        for i in idx:
            r = df.iloc[i]
            # skip if this row was used as a shot exemplar
            if subj in shot_subjs and i == 0:
                continue
            items.append({
                "id": f"mmlu/{subj}/{i}",
                "category": "mmlu",
                "prompt": (
                    fewshot +
                    f"Question: {r['question']}\n"
                    f"A. {r['choices'][0]}\nB. {r['choices'][1]}\n"
                    f"C. {r['choices'][2]}\nD. {r['choices'][3]}\n"
                    "Answer:"),
                "gold": LETTERS[int(r["answer"])],
                "max_new_tokens": 8,
            })
    # trim/extend deterministically to exactly n
    rng.shuffle(items)
    return items[:n]


def _gsm8k_gold(answer: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return m.group(1).replace(",", "") if m else ""


def build_gsm8k_items(test_pq: str, train_pq: str, n: int, seed: int) -> list[dict]:
    """Stratified by reasoning steps (tercile) x answer magnitude (tercile)."""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(test_pq).reset_index(drop=True)
    steps = df["answer"].str.count("\n")
    mag = df["answer"].apply(lambda a: (
        np.log10(abs(float(_gsm8k_gold(a)) + 1e-9)) if _gsm8k_gold(a) else 0.0))
    df = df.assign(_s=pd.qcut(steps, 3, labels=False, duplicates="drop"),
                   _m=pd.qcut(mag, 3, labels=False, duplicates="drop"))
    items: list[dict] = []
    # 8-shot CoT exemplars: fixed first 8 train rows (logged in report)
    train = pd.read_parquet(train_pq)
    shots = []
    for i in range(8):
        r = train.iloc[i]
        shots.append(f"Q: {r['question']}\nA: {r['answer']}")
    fewshot = "\n\n".join(shots) + "\n\n"
    per_cell = max(1, n // 9)
    for s in sorted(df["_s"].unique()):
        for m_ in sorted(df["_m"].unique()):
            cell = df[(df["_s"] == s) & (df["_m"] == m_)]
            if len(cell) == 0:
                continue
            idx = rng.choice(len(cell), size=min(per_cell, len(cell)), replace=False)
            for i in idx:
                r = cell.iloc[i]
                items.append({
                    "id": f"gsm8k/{r.name}",
                    "category": "gsm8k",
                    "prompt": fewshot + f"Q: {r['question']}\nA:",
                    "gold": _gsm8k_gold(r["answer"]),
                    "max_new_tokens": 256,
                })
    rng.shuffle(items)
    # Top up to n if stratification undersampled (small cells)
    if len(items) < n:
        used = {it["id"] for it in items}
        rest = df[~df.index.map(lambda ix: f"gsm8k/{ix}" in used)]
        need = n - len(items)
        if len(rest) > 0:
            idx = rng.choice(len(rest), size=min(need, len(rest)), replace=False)
            for i in idx:
                r = rest.iloc[i]
                items.append({
                    "id": f"gsm8k/{r.name}",
                    "category": "gsm8k",
                    "prompt": fewshot + f"Q: {r['question']}\nA:",
                    "gold": _gsm8k_gold(r["answer"]),
                    "max_new_tokens": 256,
                })
    rng.shuffle(items)
    return items[:n]


def build_humaneval_items(test_pq: str, n: int, seed: int) -> list[dict]:
    """Stratified by canonical-solution length tercile (difficulty proxy)."""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(test_pq).reset_index(drop=True)
    ln = df["canonical_solution"].str.len()
    df = df.assign(_d=pd.qcut(ln, 3, labels=["easy", "med", "hard"], duplicates="drop"))
    items: list[dict] = []
    per = max(1, n // 3)
    for d in ("easy", "med", "hard"):
        cell = df[df["_d"] == d]
        idx = rng.choice(len(cell), size=min(per, len(cell)), replace=False)
        for i in idx:
            r = cell.iloc[i]
            items.append({
                "id": f"humaneval/{r['task_id']}",
                "category": "humaneval",
                "prompt": r["prompt"],
                "test": r["test"],
                "entry_point": r["entry_point"],
                "max_new_tokens": 256,
            })
    rng.shuffle(items)
    return items[:n]


_FILLER = ("The quick brown fox jumps over the lazy dog. "
           "Pack my box with five dozen liquor jugs. "
           "How vexingly quick daft zebras jump! ")


def build_needle_items(n: int, seed: int,
                       contexts: tuple = (1024, 2048, 4096)) -> list[dict]:
    """Synthetic needle-in-haystack. Context sizes scaled for the Phase-B
    tiny model (spec's 4k/8k/16k target the 1.7B Phase-C model)."""
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    for i in range(n):
        ctx = contexts[i % len(contexts)]
        n_needles = int(rng.integers(1, 4))
        needles = [f"needle-{i}-{j}-" + "".join(
            rng.choice(list("abcdefghjkmnpqrstuvwxyz"), 6)) for j in range(n_needles)]
        secrets = [str(int(rng.integers(10000, 99999))) for _ in needles]
        # haystack sized to ~ctx tokens (rough: 1 token ~= 0.75 words here)
        filler_words = _FILLER.split()
        n_words = int(ctx * 0.75)
        words = (filler_words * (n_words // len(filler_words) + 1))[:n_words]
        words = list(words)
        for needle, secret in zip(needles, secrets):
            pos = int(rng.integers(0, len(words)))
            words[pos:pos] = f"The secret code {needle} is {secret}.".split()
        hay = " ".join(words)
        items.append({
            "id": f"needle/{i}",
            "category": "needle",
            "prompt": (f"Memorize the following passage, then list every secret "
                       f"code exactly.\n\n{hay}\n\nSecret codes:"),
            "gold": secrets,   # all must appear verbatim
            "max_new_tokens": 64,
        })
    return items


def build_copy_items(n: int, seed: int, tokenizer, vocab_size: int) -> list[dict]:
    """Synthetic exact-copy probes: random token strings, lengths 64-512."""
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    lengths = np.linspace(64, 512, n).astype(int)
    for i, L in enumerate(lengths):
        ids = rng.integers(0, vocab_size, size=L).tolist()
        text = tokenizer.decode(ids)
        items.append({
            "id": f"copy/{i}",
            "category": "copy",
            "prompt": f"Repeat the following text exactly, character for character:\n\n{text}\n\nRepetition:",
            "gold": text,
            "max_new_tokens": int(L * 1.2) + 16,
        })
    return items


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate_greedy(
    model, tokenizer, prompt: str, max_new_tokens: int
) -> str:
    """Greedy decoding (temperature 0), single item."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------- scoring

def score_mmlu(generation: str, gold: str) -> int:
    s = generation.strip()
    return 1 if s[:1].upper() == gold else 0


def _parse_last_number(text: str) -> str | None:
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "")


def score_gsm8k(generation: str, gold: str) -> int:
    pred = _parse_last_number(generation)
    if pred is None or not gold:
        return 0
    try:
        return 1 if abs(float(pred) - float(gold)) < 1e-6 else 0
    except ValueError:
        return 0


def score_humaneval(prompt: str, generation: str, test: str,
                    entry_point: str, timeout_s: int = 5) -> int:
    """pass@1 via sandboxed subprocess exec (5 s timeout).

    The completion continues the prompt on the same line (HumanEval
    convention: prompt ends mid-statement, completion carries its own
    indentation), so they are concatenated directly.
    """
    program = (prompt + generation + "\n" + test + f"\ncheck({entry_point})")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout_s)
        return 1 if r.returncode == 0 else 0
    except subprocess.TimeoutExpired:
        return 0
    except Exception:
        return 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def score_needle(generation: str, gold: list[str]) -> int:
    return 1 if all(g in generation for g in gold) else 0


def score_copy(generation: str, gold: str) -> int:
    return 1 if generation.strip() == gold.strip() else 0


def score_item(item: dict, generation: str, timeout_s: int = 5) -> int:
    c = item["category"]
    if c == "mmlu":
        return score_mmlu(generation, item["gold"])
    if c == "gsm8k":
        return score_gsm8k(generation, item["gold"])
    if c == "humaneval":
        return score_humaneval(item["prompt"], generation, item["test"],
                               item["entry_point"], timeout_s)
    if c == "needle":
        return score_needle(generation, item["gold"])
    if c == "copy":
        return score_copy(generation, item["gold"])
    raise ValueError(c)


def build_battery(data_dir: str, tokenizer, pcfg: ProbeConfig) -> list[dict]:
    """Assemble the full 250-item battery (deterministic)."""
    items: list[dict] = []
    items += build_mmlu_items(os.path.join(data_dir, "mmlu"), pcfg.n_mmlu, pcfg.seed)
    items += build_gsm8k_items(
        os.path.join(data_dir, "gsm8k-test.parquet"),
        os.path.join(data_dir, "gsm8k-train.parquet"),
        pcfg.n_gsm8k, pcfg.seed + 1)
    items += build_humaneval_items(
        os.path.join(data_dir, "humaneval-test.parquet"), pcfg.n_humaneval, pcfg.seed + 2)
    items += build_needle_items(pcfg.n_needle, pcfg.seed + 3)
    items += build_copy_items(pcfg.n_copy, pcfg.seed + 4, tokenizer,
                              tokenizer.vocab_size)
    assert len(items) == (pcfg.n_mmlu + pcfg.n_gsm8k + pcfg.n_humaneval
                          + pcfg.n_needle + pcfg.n_copy), len(items)
    return items
