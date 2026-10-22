"""Bit-budget optimizer — §7 of METHODOLOGY.md (spec v1.0, v1.1 §7.1/§7.4).

Multiple-choice knapsack over units: menu B = {2,3,4,5,6,8} bits mapped
to GGUF types {Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0}.

Size model (v1.1 — llama.cpp remap-aware; spec §7.1 patch):
  The v1.0 model assumed a requested tier lands uniformly
  (``size_u(b) = params_u * b / 8 * 1.02``). It does not. llama.cpp
  remaps every K-quant request per tensor, via two mechanisms in
  ``src/llama-quant.cpp`` (rev fcac190, verified in source):

  1. ``tensor_type_fallback``: a K-quant (block size 256) is incompatible
     with any tensor whose first dim (ncols) is not divisible by 256, and
     is demoted: Q2_K/Q3_K -> Q4_0, Q4_K -> Q5_0, Q5_K -> Q5_1,
     Q6_K -> Q8_0. On SmolLM2-135M every weight tensor has ncols = 576
     (576 % 256 != 0) EXCEPT ffn_down (ncols = 1536, divisible), so an
     override lands as the requested type ONLY on ffn_down; every other
     role lands demoted. E.g. Q6_K -> Q8_0 everywhere but ffn_down.
  2. The Q4_K_M mixer (``llama_tensor_get_type_impl``), which applies to
     tensors with NO manual override: attn_v / ffn_down get Q6_K on
     ``use_more_bits`` blocks (first/last 1/8 + every 3rd layer) else Q4_K;
     all other weight tensors get Q4_K; tied token_embd gets Q8_0
     (ncols % 256 != 0) else Q6_K. The fallback then demotes as above.

  ``solve_knapsack`` therefore accepts optional per-unit ``role_shapes``
  ({role: (n_params, ncols)}) and sizes each option with the remap model
  (exact, no 1.02 fudge factor). Without ``role_shapes`` it falls back to
  the v1.0 uniform model (backward compatible).

  Pipeline note: in this pipeline the base type is Q4_K_M and an optimizer
  allocation of Q4_K is emitted as NO override (both ``emit_recipe_sh``
  and the closed-loop script skip Q4_K/Q4_K_M), so those units land as
  the Q4_K_M mixer. ``expected_type_for_alloc`` encodes that.

Significance gate (pre-solve): only units with CI_low(S_u) >
max(0, CI_high(null floor)) may move; the rest are pinned at b_default=4.
Exact DP over budget discretized to MB. Novel sub-claim (iv): exact DP
over K=6 real GGUF types (firefly is binary, mlx-optiq is greedy).
"""
from __future__ import annotations

import re

import numpy as np

from .config import OptimizerConfig
from .gguf import bytes_per_param

# ---------------------------------------------------------------------------
# llama.cpp remap model (spec v1.1). Source: src/llama-quant.cpp,
# tensor_type_fallback + llama_tensor_get_type_impl (rev fcac190).
# ---------------------------------------------------------------------------

# K-quant demotion when ncols % 256 != 0 (tensor_type_fallback).
FALLBACK_TYPE: dict[str, str] = {
    "Q2_K": "Q4_0",
    "Q3_K": "Q4_0",
    "Q4_K": "Q5_0",
    "Q5_K": "Q5_1",
    "Q6_K": "Q8_0",
}

# Roles whose mixer treatment differs (Q4_K_M, no override).
_MIXER_UPGRADED_ROLES = ("attn_v", "ffn_down")


def fallback_type(requested: str, ncols: int) -> str:
    """Type that actually lands for a K-quant request on a tensor with
    first-dim ``ncols`` (tensor_type_fallback). 32-block types (Q4_0 etc.)
    and F16/F32 fit any ncols divisible by 32; for ncols % 32 != 0 the
    quantizer would fall back further to F16 (not modelled — no such
    tensor in this pipeline)."""
    from .gguf import block_size

    if ncols % block_size(requested) == 0:
        return requested
    if requested in FALLBACK_TYPE:
        return FALLBACK_TYPE[requested]
    return requested  # already a 32-block (or wider) type: stays


def use_more_bits(block_idx: int, n_layers: int) -> bool:
    """llama.cpp use_more_bits: first/last 1/8 of layers + every 3rd."""
    return (block_idx < n_layers // 8 or block_idx >= 7 * n_layers // 8
            or (block_idx - n_layers // 8) % 3 == 2)


def q4km_mix_type(role: str, block_idx: int, n_layers: int,
                  ncols: int) -> str:
    """Type the Q4_K_M mixer assigns to a NON-overridden tensor, after
    fallback. ``role`` is a gguf role name (attn_q, ..., ffn_down,
    token_embd); ``block_idx`` is None for the embedding."""
    if role == "token_embd":
        # tied-embedding branch: Q6_K if the row fits a K block else Q8_0
        base = "Q6_K" if ncols % 256 == 0 else "Q8_0"
        return fallback_type(base, ncols)
    if role in _MIXER_UPGRADED_ROLES and block_idx is not None \
            and use_more_bits(block_idx, n_layers):
        base = "Q6_K"
    else:
        base = "Q4_K"
    return fallback_type(base, ncols)


def expected_type_for_alloc(alloc_type: str, role: str, block_idx: int | None,
                            n_layers: int, ncols: int) -> str:
    """Type that lands for one tensor given the optimizer's per-unit
    allocation. Q4_K / Q4_K_M mean 'no override' -> Q4_K_M mixer;
    any other allocation is a manual override -> fallback only.

    PER-UNIT APPROXIMATION — two second-order effects (both measured on
    the Phase-B recipe file, §2.4 note) are NOT captured here; use
    ``predict_file_types`` for the exact file-level simulation:
    (1) the mixer's ``use_more_bits`` counters count only tensors that
    pass through the mixer, so a Q4_K unit's attn_v/ffn_down tier depends
    on how many EARLIER blocks were overridden (shifted blk.3/blk.12 up
    one tier in the Phase-B recipe); (2) the legacy ``output.weight``
    pattern emitted for tied embeddings regex-matches every
    ``blk.*.attn_output.weight`` tensor, so mixer-block attn_output lands
    at fallback(embed alloc) instead of the mixer Q4_K -> Q5_0
    (fixed for new recipes by ``build_patterns`` /
    ``emit_recipe_sh`` with ``tied_embeddings=True``)."""
    if alloc_type in ("Q4_K", "Q4_K_M"):
        return q4km_mix_type(role, block_idx, n_layers, ncols)
    return fallback_type(alloc_type, ncols)


def build_patterns(alloc_types: dict[str, str],
                   gguf_names_by_unit: dict[str, list[str]],
                   tied_embeddings: bool = False,
                   ) -> list[tuple[str, str]]:
    """``--tensor-type`` pattern list in emission order (sorted unit ids,
    per-unit gguf-name order — the same order closedloop / emit_recipe_sh
    emit). First matching pattern wins in llama.cpp (regex_search).

    ``tied_embeddings``: when the model's lm_head is tied to its embedding,
    the embed unit's gguf names list ``output.weight`` alongside
    ``token_embd.weight`` for the SAME tensor — but no ``output.weight``
    tensor exists in the GGUF. Emitting the bare ``output.weight`` pattern
    is a wart: as an unanchored regex it matches every
    ``blk.*.attn_output.weight``, landing mixer-block attn_output at
    fallback(embed alloc) instead of the mixer tier (P2_REPORT §6 caveat
    5). Pass ``tied_embeddings=True`` to drop the phantom pattern for
    new recipes. Default ``False`` preserves the legacy emission, which
    is what the Phase-B recipe GGUF was built with — verification of
    that artifact (``scripts/verify_recipe_sizes.py``) intentionally
    uses the default so the simulation matches the command that built
    the file.
    """
    patterns = []
    for uid, qtype in sorted(alloc_types.items()):
        if qtype in ("Q4_K_M", "Q4_K"):
            continue  # no override emitted for base-type units
        names = gguf_names_by_unit.get(uid, [uid])
        if tied_embeddings and "token_embd.weight" in names:
            names = [n for n in names if n != "output.weight"]
        for tname in names:
            patterns.append((tname, qtype))
    return patterns


def predict_file_types(patterns: list[tuple[str, str]],
                       tensors: list[dict],
                       n_layers: int) -> dict[str, str]:
    """Exact file-level type simulation for one ``llama-quantize`` run.

    ``tensors``: [{name, role, unit_id, ncols}] — order does NOT matter;
    the simulator processes them in llama.cpp's processing order
    (``weights_map`` with ``weight_name_comparer``: numeric layer index
    parsed from ``blk.%d``, ties broken by plain name compare — NOT
    GGUF header order, which is lexicographic). For each tensor: the
    first regex-matching pattern wins (manual override -> fallback
    only); otherwise the Q4_K_M mixer runs with counters that increment
    ONLY for mixer-processed attn_v / ffn_down tensors (verified: this
    reproduces the Phase-B recipe file's 211 weight-tensor types exactly,
    including the counter shift on blk.3/blk.12 and the ``output.weight``
    catch-all on mixer-block attn_output).

    Returns {tensor name: landed ggml type name}. Norm / non-weight
    tensors (role None) are returned as F32 (never quantized).
    """
    def layer_key(t: dict):
        m = re.fullmatch(r"blk\.(\d+)\..*", t["name"])
        layer = int(m.group(1)) if m else -1
        return (layer, t["name"])

    n_attn_v = sum(1 for t in tensors if t.get("role") == "attn_v")
    n_ffn_down = sum(1 for t in tensors if t.get("role") == "ffn_down")
    i_attn_v = 0
    i_ffn_down = 0
    out: dict[str, str] = {}
    for t in sorted(tensors, key=layer_key):
        name, role, uid = t["name"], t.get("role"), t.get("unit_id")
        ncols = t["ncols"]
        if role is None:
            out[name] = "F32"
            continue
        manual = None
        for pat, qtype in patterns:
            if re.search(pat, name):
                manual = qtype
                break
        if manual is not None:
            out[name] = fallback_type(manual, ncols)
            continue
        # mixer path (Q4_K_M)
        if role == "token_embd":
            base = "Q6_K" if ncols % 256 == 0 else "Q8_0"
            out[name] = fallback_type(base, ncols)
        elif role in _MIXER_UPGRADED_ROLES:
            if role == "attn_v":
                upgraded = use_more_bits(i_attn_v, n_attn_v)
                i_attn_v += 1
            else:
                upgraded = use_more_bits(i_ffn_down, n_ffn_down)
                i_ffn_down += 1
            out[name] = fallback_type("Q6_K" if upgraded else "Q4_K", ncols)
        else:
            out[name] = fallback_type("Q4_K", ncols)
    return out


def role_bytes(role_shapes: dict[str, tuple[int, int]], requested: str,
               block_idx: int | None, n_layers: int) -> float:
    """Predicted stored bytes for one unit's weight tensors under the
    requested allocation, using the remap model. ``role_shapes`` maps
    gguf role -> (n_params, ncols). Norm tensors (F32) are NOT included
    here; add them separately (they are never quantized)."""
    total = 0.0
    for role, (n_params, ncols) in role_shapes.items():
        landed = expected_type_for_alloc(requested, role, block_idx,
                                         n_layers, ncols)
        total += n_params * bytes_per_param(landed)
    return total


def block_index(unit_id: str) -> int | None:
    """'blk.17' -> 17; 'embed'/'lm_head' -> None."""
    m = re.fullmatch(r"blk\.(\d+)", unit_id)
    return int(m.group(1)) if m else None


def predicted_size_bytes(alloc_types: dict[str, str],
                         role_shapes_by_unit: dict[str, dict[str, tuple[int, int]]],
                         n_layers: int,
                         norm_bytes: float = 0.0) -> float:
    """Predicted total weight-tensor bytes for an allocation under the
    remap model. ``norm_bytes`` adds the (unquantized, F32) norm tensors."""
    total = norm_bytes
    for uid, alloc in alloc_types.items():
        shapes = role_shapes_by_unit.get(uid)
        if shapes is None:
            continue
        total += role_bytes(shapes, alloc, block_index(uid), n_layers)
    return total


def predicted_loss(c_hat: float, alpha: float, b: int, b_ref: int = 4) -> float:
    # never extrapolate the fitted slope below b=3 — clamp (§7.2)
    b_eff = max(b, 3)
    return c_hat * 2.0 ** (-alpha * (b_eff - b_ref))


def unit_size_mb(n_params: int, b: int, overhead: float = 1.02) -> float:
    return n_params * b / 8 * overhead / 1e6


def _unit_size_mb(unit: dict, b: int, ocfg: OptimizerConfig,
                 n_layers: int) -> float:
    """Size of one unit at menu option ``b`` (MB). With ``role_shapes``
    on the unit, uses the remap model (exact); otherwise the v1.0
    uniform model (``params * b / 8 * 1.02``)."""
    shapes = unit.get("role_shapes")
    if shapes:
        alloc = ocfg.gguf_types[b]
        return role_bytes(shapes, alloc, block_index(unit["id"]),
                          n_layers) / 1e6
    return unit_size_mb(unit["n_params"], b, ocfg.overhead)


def solve_knapsack(
    units: list,                 # list of dicts: id, n_params, c_hat, alpha, gated
    budget_mb: float,
    ocfg: OptimizerConfig | None = None,
) -> dict:
    """Exact DP multiple-choice knapsack. Returns {unit_id: bits} + stats.

    Budget discretized into bins (spec §7.3: 1 MB bins at 1.7B scale);
    bin width = 1 MB when the budget >= 500 MB, else budget/1000 so the
    constraint still binds on tiny models.
    """
    ocfg = ocfg or OptimizerConfig()
    menu = list(ocfg.menu_bits)
    n_layers = sum(1 for u in units if block_index(u["id"]) is not None)
    # Bin width: 1 MB at 1.7B scale (spec §7.3); budget/2000 for smaller
    # budgets so bin-rounding cannot overshoot the budget (a 69-bin
    # discretization overshot by 6.3 MB on the 68 MB Phase-B budget).
    if budget_mb >= 500:
        n_bins = min(2000, max(100, int(round(budget_mb))))
    else:
        n_bins = 2000
    bin_mb = budget_mb / n_bins
    Smax = n_bins
    U = len(units)
    INF = float("inf")
    dp = np.full(Smax + 1, INF)
    dp[0] = 0.0
    # choice[u][s] = bits chosen for unit u at budget s (int8, -1 = unset)
    choice = np.full((U, Smax + 1), -1, dtype=np.int8)
    prev = np.full((U, Smax + 1), -1, dtype=np.int32)

    for u, unit in enumerate(units):
        opts = []
        for b in menu:
            if not unit["gated"] and b != ocfg.default_bits:
                continue  # pinned at default
            size_mb = _unit_size_mb(unit, b, ocfg, n_layers)
            loss = predicted_loss(unit["c_hat"], unit["alpha"], b)
            opts.append((b, size_mb, loss))
        newdp = np.full(Smax + 1, INF)
        for s in range(Smax + 1):
            if dp[s] == INF:
                continue
            for b, size_mb, loss in opts:
                s2 = s + int(round(size_mb / bin_mb))
                if s2 <= Smax and dp[s] + loss < newdp[s2]:
                    newdp[s2] = dp[s] + loss
                    choice[u, s2] = menu.index(b)
                    prev[u, s2] = s
        dp = newdp

    best_s = int(np.argmin(dp))
    alloc: dict[str, int] = {}
    s = best_s
    for u in range(U - 1, -1, -1):
        bi = choice[u, s]
        b = menu[bi] if bi >= 0 else ocfg.default_bits
        alloc[units[u]["id"]] = b
        s = prev[u, s] if prev[u, s] >= 0 else 0
    # Bin-rounding repair: the DP constrains ROUNDED bin counts, so the
    # true total can exceed the budget by up to ~U/2 bins (observed
    # 103.61 vs 103.53 MB on a 31-unit SmolLM2 CLI run). While over
    # budget, downgrade the unit with the smallest loss increase per MB
    # saved to its next-lower menu option (ungated units are pinned and
    # never move). No-op when the DP solution already fits (e.g. the
    # stored Phase-B solution: used == budget).
    def _total_mb(a: dict) -> float:
        return sum(_unit_size_mb(units[u], a[units[u]["id"]], ocfg, n_layers)
                   for u in range(U))

    guard = 0
    while _total_mb(alloc) > budget_mb + 1e-9:
        guard += 1
        if guard > 100 * U + 10:
            raise ValueError("knapsack repair did not converge")
        best: tuple[float, int, int] | None = None  # (ratio, unit, new_b)
        for u in range(U):
            unit = units[u]
            cur = alloc[unit["id"]]
            if not unit["gated"]:
                continue  # pinned at default
            lower = [b for b in menu if b < cur]
            if not lower:
                continue
            nb = max(lower)  # immediate next-lower option
            d_loss = (predicted_loss(unit["c_hat"], unit["alpha"], nb)
                      - predicted_loss(unit["c_hat"], unit["alpha"], cur))
            d_mb = (_unit_size_mb(unit, cur, ocfg, n_layers)
                    - _unit_size_mb(unit, nb, ocfg, n_layers))
            if d_mb <= 0:
                continue
            ratio = d_loss / d_mb
            if best is None or ratio < best[0]:
                best = (ratio, u, nb)
        if best is None:
            raise ValueError(
                f"knapsack: budget {budget_mb} MB infeasible after "
                "bin-rounding repair (no unit can downgrade)")
        alloc[units[best[1]]["id"]] = best[2]
    total_loss = float(sum(predicted_loss(units[u]["c_hat"],
                                          units[u]["alpha"],
                                          alloc[units[u]["id"]])
                           for u in range(U)))
    total_mb = _total_mb(alloc)
    return {
        "alloc_bits": alloc,
        "alloc_types": {uid: ocfg.gguf_types[b] for uid, b in alloc.items()},
        "predicted_loss": total_loss,
        "budget_mb": budget_mb,
        "used_mb": total_mb,
        "bin_mb": bin_mb,
        "n_bins": n_bins,
        "solver": "exact_dp",
    }


def emit_recipe_sh(
    alloc_types: dict[str, str],
    gguf_names: dict[str, list[str]],
    model_f16_gguf: str = "model-f16.gguf",
    out_gguf: str = "model-bitblame.gguf",
    base_type: str = "Q4_K_M",
    tied_embeddings: bool = False,
) -> str:
    """Compile the optimizer output to a copy-paste recipe.sh (§7.4).

    ``tied_embeddings``: pass True when the model's lm_head is tied to
    its embedding — the phantom ``output.weight`` pattern is then dropped
    (see ``build_patterns``: the bare pattern regex-matches every
    ``blk.*.attn_output.weight``). Default False preserves the legacy
    emission that built the Phase-B artifact.
    """
    lines = [
        "#!/bin/sh",
        "# BitBlame recipe — measured-sensitivity-driven GGUF quantization",
        "# Generated per METHODOLOGY.md §7.4 (spec v1.0).",
        "# Unmentioned tensors fall back to the base type.",
        "set -e",
        "",
        "# 1) importance matrix on the calibration mix (§5.1)",
        "llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat",
        "",
        "# 2) quantize with per-tensor overrides (measured allocation)",
        "llama-quantize --imatrix imatrix.dat \\",
    ]
    overrides = []
    for uid, qtype in sorted(alloc_types.items()):
        if qtype in (base_type, "Q4_K"):
            continue
        names = gguf_names.get(uid, [uid])
        if tied_embeddings and "token_embd.weight" in names:
            names = [n for n in names if n != "output.weight"]
        for tname in names:
            # escape regex-significant chars minimally; llama.cpp takes regex
            overrides.append(f"  --tensor-type '{tname}={qtype}' \\")
    lines += overrides
    lines += [
        f"  {model_f16_gguf} {out_gguf} {base_type}",
        "",
        'echo "recipe done -> %s"' % out_gguf,
    ]
    return "\n".join(lines) + "\n"
