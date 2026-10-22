"""Phase B — Stage 3/5 (real toolchain): probe battery via llama-server.

Runs the 250-item battery (§5.2) against GGUF artifacts through the REAL
llama.cpp toolchain:
  * smollm2-f16.gguf     — FP16 reference (M0)
  * smollm2-q4km.gguf     — uniform Q4_K_M baseline (M_base)
  * smollm2-recipe.gguf   — recipe.sh output (M_recipe)

Greedy decoding (temperature 0), identical prompts/order per variant
(paired design). llama-server is started once per variant on 127.0.0.1
and shut down afterwards; /completion with cache_prompt=true reuses the
GSM8K 8-shot prefix KV cache across items.

Writes hidden_files/phaseB_gguf_<variant>.json (per-item records).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/home/hatch/workspace/bitblame/src")

from bitblame import probes
from bitblame.config import ProbeConfig

ROOT = "/home/hatch/workspace/bitblame"
HID = f"{ROOT}/hidden_files"
MODELS = f"{ROOT}/data/models"
BIN = f"{ROOT}/third_party/llama.cpp/build/bin"
LD = {"LD_LIBRARY_PATH": BIN + ":" + os.environ.get("LD_LIBRARY_PATH", "")}

VARIANTS = {
    "f16": "SmolLM2-135M-f16.gguf",
    "q4km": "SmolLM2-135M-q4km.gguf",
    "recipe": "SmolLM2-135M-recipe.gguf",
}


def start_server(gguf: str, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(LD)
    p = subprocess.Popen(
        [f"{BIN}/llama-server", "-m", f"{MODELS}/{gguf}",
         "--port", str(port), "-c", "8192", "--log-disable"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    # wait for readiness
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                   timeout=5)
            return p
        except Exception:
            time.sleep(2)
    p.kill()
    raise RuntimeError(f"server for {gguf} did not become ready")


def complete(port: int, prompt: str, n_predict: int,
             timeout_s: int = 300) -> tuple[str, dict]:
    body = json.dumps({
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
        "cache_prompt": True, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        d = json.load(r)
    dt = time.time() - t0
    timings = d.get("timings", {})
    return d.get("content", ""), {
        "seconds": round(dt, 1),
        "prefill_tps": round(timings.get("prompt_per_second", 0), 1),
        "gen_tps": round(timings.get("predicted_per_second", 0), 1),
    }


def run_variant(items: list[dict], variant: str, pcfg: ProbeConfig,
                port: int) -> list[dict]:
    out_path = f"{HID}/phaseB_gguf_{variant}.json"
    done = {}
    if os.path.exists(out_path):
        done = {r["id"]: r for r in json.load(open(out_path))}
        print(f"resuming {variant}: {len(done)}/{len(items)}", flush=True)
    server = start_server(VARIANTS[variant], port)
    try:
        for i, item in enumerate(items):
            if item["id"] in done:
                continue
            gen, tm = complete(port, item["prompt"], item["max_new_tokens"])
            score = probes.score_item(item, gen,
                                      timeout_s=pcfg.humaneval_timeout_s)
            done[item["id"]] = {
                "id": item["id"], "category": item["category"],
                "generation": gen, "score": score, **tm,
            }
            if (i + 1) % 25 == 0 or i == len(items) - 1:
                with open(out_path, "w") as f:
                    json.dump(list(done.values()), f)
                print(f"[{variant}] {len(done)}/{len(items)}", flush=True)
    finally:
        with open(out_path, "w") as f:
            json.dump(list(done.values()), f)
        server.terminate()
    return list(done.values())


def main() -> None:
    pcfg = ProbeConfig()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODELS}/SmolLM2-135M", local_files_only=True)
    items = probes.build_battery(f"{ROOT}/data", tokenizer, pcfg)
    for it in items:
        if it["category"] == "gsm8k":
            it["max_new_tokens"] = 160
    with open(f"{HID}/phaseB_battery.json", "w") as f:
        json.dump([{k: v for k, v in it.items() if k != "prompt"}
                   for it in items], f, indent=2)
    print(f"battery: {len(items)} items", flush=True)
    for v, (variant, port) in enumerate(
            [("f16", 18081), ("q4km", 18082), ("recipe", 18083)]):
        if not os.path.exists(f"{MODELS}/{VARIANTS[variant]}"):
            print(f"SKIP {variant}: {VARIANTS[variant]} missing", flush=True)
            continue
        t0 = time.time()
        res = run_variant(items, variant, pcfg, port)
        acc = sum(r["score"] for r in res) / len(res)
        print(f"[{variant}] done in {(time.time()-t0)/60:.1f}m  "
              f"accuracy={acc:.3f}", flush=True)
    print("GGUF PROBES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
