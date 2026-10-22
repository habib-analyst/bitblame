"""llama.cpp imatrix parsing + agreement diagnostic — METHODOLOGY.md §7.4.2.

``llama-imatrix`` writes a GGUF container (``general.type == "imatrix"``)
with four tensors per collected weight tensor::

    <name>.in_sum2   [ncols] F32   accumulated squared inputs per column
    <name>.counts    [1]     F32   number of tokens accumulated
    <name>.in_sum    [ncols] F32   accumulated inputs per column
    <name>.stats     [12]    F32   [sum_sq, mean, elements, std_deviation,
                                     skewness, kurtosis, gain, h_norm,
                                     l2_dist, cossim, pearson, covariance]

Per-tensor importance (primary): ``mean(in_sum2)`` — the average
per-column input activation energy, i.e. what llama.cpp uses to weight
that tensor's quantization error, normalized by width so tensors with
different ``ncols`` (e.g. ``ffn_down`` at 1536 vs the rest at 576) are
comparable. ``stats[0]`` (``sum_sq``) is the same quantity summed over
columns instead of averaged; it is reported as a robustness variant.

Per-unit importance: mean of the unit's tensors' importance (a block's
7 weight tensors; the embedding has no imatrix entry — ``llama-imatrix``
collects matmul weights only — so ``embed`` is excluded from the
correlation, and the comparison is over the 30 block units).

Agreement: Spearman rank correlation between per-unit imatrix importance
and BitBlame's blame score ``S_u`` (spec v1.1 §7.4.2, corroboration bar
rho >= 0.5, reported either way — not blocking).
"""
from __future__ import annotations

import re

import numpy as np
from scipy import stats as _ss

from . import gguf as _gguf

# llama.cpp imatrix stats layout (see module docstring).
STATS_FIELDS = ("sum_sq", "mean", "elements", "std_deviation", "skewness",
                "kurtosis", "gain", "h_norm", "l2_dist", "cossim",
                "pearson", "covariance")

_TENSOR_SUFFIXES = ("in_sum2", "counts", "in_sum", "stats")


def read_imatrix(path: str) -> dict:
    """Parse an ``llama-imatrix`` .dat file (GGUF container).

    Returns ``{"metadata": ..., "tensors": {base_name: {"in_sum2": ...,
    "counts": float, "in_sum": ..., "stats": {field: value}}}}`` where
    ``base_name`` is the weight-tensor name (e.g. ``blk.3.ffn_up.weight``)
    and the arrays are float64 numpy.
    """
    header = _gguf.read_gguf_header(path)
    if header["metadata"].get("general.type") != "imatrix":
        raise _gguf.GGUFError(f"{path}: general.type is "
                              f"{header['metadata'].get('general.type')!r}, "
                              "not 'imatrix'")
    with open(path, "rb") as f:
        buf = f.read()
    data_start = header["data_start"]
    out: dict[str, dict] = {}
    for name, t in header["tensors"].items():
        m = re.fullmatch(r"(.*)\.(in_sum2|counts|in_sum|stats)$", name)
        if not m:
            raise _gguf.GGUFError(f"{path}: unexpected imatrix tensor {name!r}")
        base, kind = m.group(1), m.group(2)
        if t["type"] != "F32":
            raise _gguf.GGUFError(f"{path}: tensor {name}: type {t['type']}, "
                                  "expected F32")
        vals = np.frombuffer(buf, dtype="<f4", count=t["n_elts"],
                             offset=data_start + t["offset"]).astype(np.float64)
        slot = out.setdefault(base, {})
        if kind == "counts":
            slot[kind] = float(vals[0]) if vals.size else float("nan")
        elif kind == "stats":
            slot[kind] = {f: float(v) for f, v in zip(STATS_FIELDS, vals)}
        else:
            slot[kind] = vals
    for base, slot in out.items():
        missing = [k for k in _TENSOR_SUFFIXES if k not in slot]
        if missing:
            raise _gguf.GGUFError(f"{path}: tensor {base}: missing {missing}")
    return {"metadata": header["metadata"], "tensors": out}


def tensor_importance(slot: dict, variant: str = "mean_in_sum2") -> float:
    """Scalar importance for one imatrix tensor entry.

    ``mean_in_sum2`` (primary): mean of ``in_sum2`` over input columns.
    ``sum_sq`` (robustness variant): the ``stats[0]`` total input energy.
    """
    if variant == "mean_in_sum2":
        return float(np.mean(slot["in_sum2"]))
    if variant == "sum_sq":
        return float(slot["stats"]["sum_sq"])
    raise ValueError(f"unknown imatrix importance variant {variant!r}")


def unit_importance(imatrix: dict, variant: str = "mean_in_sum2") -> dict[str, float]:
    """Per-unit imatrix importance: mean over the unit's weight tensors.

    Units are ``blk.<i>`` (7 tensors each). Tensors that do not belong to
    a block (none in current imatrix files — embeddings are not
    collected) are ignored.
    """
    per_unit: dict[str, list[float]] = {}
    for base, slot in imatrix["tensors"].items():
        m = re.fullmatch(r"blk\.(\d+)\..*\.weight", base)
        if not m:
            continue
        per_unit.setdefault(f"blk.{m.group(1)}", []).append(
            tensor_importance(slot, variant))
    return {u: float(np.mean(v)) for u, v in sorted(per_unit.items())}


def agreement(imatrix: dict, blame: dict[str, float],
              variant: str = "mean_in_sum2") -> dict:
    """Spearman agreement between per-unit imatrix importance and ``S_u``.

    ``blame`` maps unit id -> blame score. Only units present in BOTH
    inputs are compared (the embedding is excluded: no imatrix entry).
    Returns rho, p-value, n, the bar verdict, and the per-unit table.
    """
    imp = unit_importance(imatrix, variant)
    common = sorted(set(imp) & set(blame))
    xs = np.array([imp[u] for u in common], dtype=np.float64)
    ys = np.array([blame[u] for u in common], dtype=np.float64)
    res = _ss.spearmanr(xs, ys)
    # rank disagreement per unit (|rank_imatrix - rank_blame|), for naming
    # diverging tensors when rho is below the bar (§7.4.2).
    r_imp = _ss.rankdata(-xs)
    r_blame = _ss.rankdata(-ys)
    table = [
        {"unit": u, "imatrix_importance": float(imp[u]),
         "S_u": float(blame[u]),
         "rank_imatrix": int(r_imp[i]), "rank_blame": int(r_blame[i]),
         "rank_gap": int(abs(r_imp[i] - r_blame[i]))}
        for i, u in enumerate(common)
    ]
    rho = float(res.statistic)
    return {
        "variant": variant,
        "rho": rho,
        "p_value": float(res.pvalue),
        "n_units": len(common),
        "units": [u for u in common],
        "bar": 0.5,
        "meets_bar": bool(rho >= 0.5),
        "per_unit": table,
    }
