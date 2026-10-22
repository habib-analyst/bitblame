"""Phase B — Stage 5 (real toolchain): the closed loop through llama.cpp.

  1. calib.txt from the Phase-B calibration sequences (decoded text).
  2. llama-imatrix on the F16 GGUF -> imatrix.dat (CPU).
  3. Baseline: llama-quantize --imatrix ... f16.gguf q4km.gguf Q4_K_M
  4. Recipe:   llama-quantize --imatrix ... [--tensor-type overrides from
     phaseB_analysis.json] f16.gguf recipe.gguf Q4_K_M
  5. Verify: full type simulation (src/bitblame/optimizer.predict_
     file_types) vs a pure-Python GGUF header read
     (src/bitblame/gguf.py). The old step 5 shelled out to
     `llama-gguf <file> r`, whose hardcoded self-test asserts
     data[0] == 100.0 and aborts on any real model file — the harness
     exited 1 on a healthy artifact. Never use `llama-gguf r` here.

Novel sub-claim (v): the loop runs through the actual GGUF toolchain.
llama.cpp revision is recorded in the report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
MODELS = f"{ROOT}/data/models"
LLAMA = f"{ROOT}/third_party/llama.cpp"
BIN = f"{LLAMA}/build/bin"


def sh(cmd: list[str], **kw) -> str:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:], flush=True)
        print(r.stderr[-2000:], flush=True)
        raise RuntimeError(f"command failed: {cmd[0]}")
    return r.stdout


def main() -> None:
    os.makedirs(HID, exist_ok=True)
    # 1) calib.txt from the calibration sequences
    from transformers import AutoTokenizer
    sys.path.insert(0, f"{ROOT}/scripts")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase_b_scan", f"{ROOT}/scripts/phase_b_scan.py")
    # (avoid importing: rebuild sequences directly from calib.json)
    calib = json.load(open(f"{HID}/phaseB_calib.json"))
    import pandas as pd
    df = pd.read_parquet(f"{ROOT}/data/fineweb-edu-docs.parquet")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODELS}/SmolLM2-135M", local_files_only=True)
    with open(f"{HID}/calib.txt", "w") as f:
        for m in calib["sequences"]:
            ids = tokenizer(str(df.iloc[m["doc_row"]]["text"]),
                            truncation=False)["input_ids"][:calib["seq_len"]]
            f.write(tokenizer.decode(ids) + "\n")
    print("calib.txt written", flush=True)

    rev = subprocess.run(["git", "-C", LLAMA, "rev-parse", "--short",
                          "HEAD"], capture_output=True, text=True).stdout.strip()

    # 2) imatrix
    f16 = f"{MODELS}/SmolLM2-135M-f16.gguf"
    if not os.path.exists(f"{HID}/imatrix.dat"):
        sh([f"{BIN}/llama-imatrix", "-m", f16, "-f", f"{HID}/calib.txt",
            "-o", f"{HID}/imatrix.dat", "-c", "512", "-b", "512"])
    else:
        print("imatrix.dat exists, skipping", flush=True)

    # 3) baseline Q4_K_M
    q4km = f"{MODELS}/SmolLM2-135M-q4km.gguf"
    if not os.path.exists(q4km):
        sh([f"{BIN}/llama-quantize", "--imatrix", f"{HID}/imatrix.dat",
            f16, q4km, "Q4_K_M"])
    else:
        print("q4km exists, skipping", flush=True)

    # 4) recipe quantize with --tensor-type overrides from the optimizer
    analysis = json.load(open(f"{HID}/phaseB_analysis.json"))
    alloc_types = analysis["optimizer"]["alloc_types"]
    gguf_names = {}
    for row in analysis["unit_rows"]:
        uid = row["id"]
        # reconstruct gguf names: blk.i -> blk.i.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight
        if uid.startswith("blk."):
            i = uid.split(".")[1]
            gguf_names[uid] = [
                f"blk.{i}.{r}.weight" for r in
                ("attn_q", "attn_k", "attn_v", "attn_output",
                 "ffn_gate", "ffn_up", "ffn_down")]
        elif uid == "embed":
            gguf_names[uid] = ["token_embd.weight", "output.weight"]
    cmd = [f"{BIN}/llama-quantize", "--imatrix", f"{HID}/imatrix.dat"]
    for uid, qtype in sorted(alloc_types.items()):
        if qtype in ("Q4_K_M", "Q4_K"):
            continue
        for tname in gguf_names[uid]:
            cmd += ["--tensor-type", f"{tname}={qtype}"]
    recipe_gguf = f"{MODELS}/SmolLM2-135M-recipe.gguf"
    cmd += [f16, recipe_gguf, "Q4_K_M"]
    sh(cmd)

    # 5) verify per-tensor types in the recipe artifact — GGUF header
    # read (src/bitblame/gguf.py), NOT `llama-gguf r`: the example tool's
    # `r` mode runs a hardcoded self-test (gguf_ex_read_1 expects
    # data[0] == 100.0) and GGML_ASSERT-aborts on ANY real model file.
    # That is what made the old harness exit 1 on a healthy artifact.
    # Verification = full file-level simulation (optimizer.predict_
    # file_types: --tensor-type regex patterns in emission order +
    # Q4_K_M mixer with mixer-only counters) vs the actual header.
    import re as _re
    sys.path.insert(0, f"{ROOT}/src")
    from bitblame import gguf as _gguf, optimizer as _opt

    f16_hdr = _gguf.read_gguf_header(f16)
    recipe_hdr = _gguf.read_gguf_header(recipe_gguf)
    tensors = []
    for _name, _t in f16_hdr["tensors"].items():
        _m = _re.fullmatch(
            r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight",
            _name)
        if _m:
            _uid, _role = f"blk.{_m.group(1)}", _m.group(2)
        elif _name == "token_embd.weight":
            _uid, _role = "embed", "token_embd"
        elif _name == "output.weight":
            _uid, _role = "embed", "output"
        else:
            _uid, _role = None, None
        tensors.append({"name": _name, "unit_id": _uid, "role": _role,
                        "ncols": _t["dims"][0], "n_elts": _t["n_elts"]})
    _gguf_names = {f"blk.{i}": [f"blk.{i}.{r}.weight" for r in
                   ("attn_q", "attn_k", "attn_v", "attn_output",
                    "ffn_gate", "ffn_up", "ffn_down")]
                   for i in range(30)}
    _gguf_names["embed"] = ["token_embd.weight", "output.weight"]
    _patterns = _opt.build_patterns(alloc_types, _gguf_names)
    _sim = _opt.predict_file_types(_patterns, tensors, 30)
    actual = {n: t["type"] for n, t in recipe_hdr["tensors"].items()}
    mismatches = [{"tensor": n, "want": _sim[n], "got": actual[n]}
                  for n in _sim if _sim[n] != actual.get(n)]
    with open(f"{HID}/tensor_types_recipe.json", "w") as f:
        json.dump(actual, f, indent=2, sort_keys=True)
    checked = len(_sim)
    if mismatches:
        raise RuntimeError(
            f"tensor-type verification FAILED: {len(mismatches)} "
            f"mismatches of {checked}: {mismatches[:5]}")
    with open(f"{HID}/closedloop.json", "w") as f:
        json.dump({
            "llama_cpp_rev": rev,
            "recipe_alloc": alloc_types,
            "tensors_checked": checked,
            "artifacts": {"f16": f16, "q4km": q4km, "recipe": recipe_gguf},
            "tensor_type_mismatches": mismatches,
            "verification": "gguf_header_read (src/bitblame/gguf.py) + "
                            "full type simulation "
                            "(optimizer.predict_file_types); "
                            "llama-gguf r self-test NOT used (it asserts "
                            "data[0]==100.0 on any real file)",
            "sizes": {k: os.path.getsize(v) for k, v in
                      {"f16": f16, "q4km": q4km,
                       "recipe": recipe_gguf}.items()},
        }, f, indent=2)
    print(f"verified {checked} tensors, {len(mismatches)} mismatches "
          f"(header read, no llama-gguf self-test)",
          flush=True)
    print("CLOSED LOOP (quantize) COMPLETE; rev", rev, flush=True)


if __name__ == "__main__":
    main()
