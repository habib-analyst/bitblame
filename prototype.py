"""BitBlame core-loop prototype — pure NumPy, CPU.

Builds a tiny randomly-initialized transformer and implements the core
measurement loop:

  (a) per-layer weight perturbation calibrated to a FAKE quantization
      step size (uniform INT8 step per weight matrix),
  (b) forward pass over a handful of calibration token sequences,
  (c) KL-divergence between unperturbed and perturbed output logits,
  (d) rank layers by sensitivity.

This validates the PLUMBING ONLY, not real quantization behavior:
perturbations are synthetic uniform noise, not actual quantized weights,
and the model is random, not trained.
"""

import json
import time

import numpy as np

# ---------------------------------------------------------------- config
SEED = 7
V = 128      # vocab size
D = 64       # hidden size
H = 4        # attention heads
L = 4        # layers
DFF = 128    # feed-forward dim
SEQ = 32     # sequence length
NCAL = 8     # calibration sequences
BITS = 8     # fake quantization bit-width (INT8)
EPS = 1e-9


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def split_heads(t, B, T):
    return t.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)


def merge_heads(t, B, T):
    return t.transpose(0, 2, 1, 3).reshape(B, T, D)


def build_params(rng):
    p = {}
    p["embed"] = rng.normal(0, 1, (V, D)) * 0.1
    for i in range(L):
        k = f"L{i}."
        p[k + "ln1"] = np.ones(D)
        p[k + "wqkv"] = rng.normal(0, 1, (D, 3 * D)) * np.sqrt(2 / (D + 3 * D))
        p[k + "wo"] = rng.normal(0, 1, (D, D)) * np.sqrt(2 / (2 * D))
        p[k + "ln2"] = np.ones(D)
        p[k + "w1"] = rng.normal(0, 1, (D, DFF)) * np.sqrt(2 / (D + DFF))
        p[k + "w2"] = rng.normal(0, 1, (DFF, D)) * np.sqrt(2 / (DFF + D))
    p["lnf"] = np.ones(D)
    p["head"] = rng.normal(0, 1, (D, V)) * 0.05
    return p


def forward(params, ids):
    B, T = ids.shape
    x = params["embed"][ids]
    for i in range(L):
        k = f"L{i}."
        h = rmsnorm(x, params[k + "ln1"])
        qkv = h @ params[k + "wqkv"]
        q, kv1, kv2 = np.split(qkv, 3, axis=-1)
        qh, kh, vh = split_heads(q, B, T), split_heads(kv1, B, T), split_heads(kv2, B, T)
        att = softmax(qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(D // H)) @ vh
        x = x + merge_heads(att, B, T) @ params[k + "wo"]
        h2 = rmsnorm(x, params[k + "ln2"])
        x = x + (np.maximum(h2 @ params[k + "w1"], 0) @ params[k + "w2"])
    return rmsnorm(x, params["lnf"]) @ params["head"]  # (B, T, V)


def quant_step(w, bits=BITS):
    """Fake uniform-quantization step size for one weight matrix."""
    return 2.0 * float(np.abs(w).max()) / (2 ** bits - 1)


def perturb_layer(params, layer_idx, rng, scale=1.0):
    """Copy params, add uniform noise in [-delta/2, delta/2] to each weight
    matrix of the target layer. delta is the fake INT8 step per matrix.
    LayerNorm scales are left untouched (kept at high precision in practice)."""
    new = dict(params)
    k = f"L{layer_idx}."
    for name in ("wqkv", "wo", "w1", "w2"):
        w = new[k + name]
        delta = quant_step(w)
        noise = rng.uniform(-0.5, 0.5, w.shape) * delta * scale
        new[k + name] = w + noise
    return new


def kl_div(logits_p, logits_q):
    p = softmax(logits_p)
    q = np.clip(softmax(logits_q), EPS, 1.0)
    return float((p * (np.log(np.clip(p, EPS, 1.0)) - np.log(q))).sum(-1).mean())


def run(seed):
    rng = np.random.default_rng(seed)
    params = build_params(rng)
    calib = rng.integers(0, V, (NCAL, SEQ))
    base_logits = forward(params, calib)

    # sanity: identical params -> KL ~ 0
    control = kl_div(base_logits, forward(dict(params), calib))

    prng = np.random.default_rng(seed + 1000)
    sens, steps = {}, {}
    for i in range(L):
        pert = perturb_layer(params, i, prng, scale=1.0)
        steps[f"L{i}"] = {n: quant_step(pert[f"L{i}.{n}"]) for n in ("wqkv", "wo", "w1", "w2")}
        sens[f"L{i}"] = kl_div(base_logits, forward(pert, calib))
    ranking = sorted(sens, key=sens.get, reverse=True)
    return control, sens, ranking, steps


def main():
    t0 = time.time()
    results = {}
    for seed in (7, 1234):
        control, sens, ranking, steps = run(seed)
        results[str(seed)] = {"control_kl": control, "sens": sens, "ranking": ranking}
        print(f"\nseed={seed}  control KL (no perturbation) = {control:.3e}")
        print("layer   KL-divergence   rank")
        for r, layer in enumerate(ranking, 1):
            print(f"  {layer}    {sens[layer]:.6f}      {r}")
    dt = time.time() - t0
    results["runtime_s"] = dt
    results["config"] = {"V": V, "D": D, "H": H, "L": L, "DFF": DFF,
                         "SEQ": SEQ, "NCAL": NCAL, "BITS": BITS}
    with open("/home/hatch/workspace/bitblame/prototype_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nruntime: {dt:.1f}s  (results -> prototype_results.json)")

    # ASCII bar chart (robust: no matplotlib needed)
    print("\nSensitivity ranking (seed 7) — KL divergence, larger = more sensitive:")
    s7 = results["7"]["sens"]
    mx = max(s7.values())
    for layer in results["7"]["ranking"]:
        bar = "#" * int(40 * s7[layer] / mx)
        print(f"  {layer}  {s7[layer]:.2e}  {bar}")

    # bar chart of seed-7 ranking (optional; skipped if matplotlib is broken)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # e.g. numpy 2.x ABI vs old system matplotlib
        print(f"chart skipped (matplotlib unavailable: {e})")
        return
    s7 = results["7"]["sens"]
    order = results["7"]["ranking"]
    vals = [s7[k] for k in order]
    plt.figure(figsize=(6, 4))
    plt.bar(order, vals, color="#4C78A8")
    plt.ylabel("KL divergence (unperturbed || perturbed)")
    plt.xlabel("layer")
    plt.title("BitBlame prototype: per-layer sensitivity ranking (seed 7)")
    plt.tight_layout()
    plt.savefig("/home/hatch/workspace/bitblame/sensitivity_prototype.png", dpi=120)
    print("chart -> sensitivity_prototype.png")


if __name__ == "__main__":
    main()
