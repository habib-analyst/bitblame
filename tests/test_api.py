"""Public-API tests — sensitivity.rank_layers / sensitivity_report on
the tiny random transformer (CPU, seconds)."""
import numpy as np
import torch

from bitblame import sensitivity as sens
from bitblame.modeling import TinyTransformer


def _tiny_batches():
    rng = np.random.default_rng(0)
    ids = torch.tensor(rng.integers(0, 256, size=(2, 16)), dtype=torch.long)
    return [ids]


def test_rank_layers_tiny():
    model = TinyTransformer(seed=3)
    rows = sens.rank_layers(model, _tiny_batches(), bits=4, n_draws=1)
    assert len(rows) == 6  # 4 blocks + embed + lm_head (untied)
    assert all(r["S"] >= 0 for r in rows)
    assert [r["S"] for r in rows] == sorted(
        [r["S"] for r in rows], reverse=True)
    assert {r["id"] for r in rows} == {
        "blk.0", "blk.1", "blk.2", "blk.3", "embed", "lm_head"}


def test_sensitivity_report_tiny():
    model = TinyTransformer(seed=3)
    rep = sens.sensitivity_report(model, _tiny_batches(), bits=4, n_draws=1,
                                  null_bits=8, n_null_draws=1,
                                  bootstrap_B=100, top_k=4)
    assert len(rep["units"]) == 6
    for u in rep["units"]:
        assert u["ci_low"] <= u["S"] <= u["ci_high"]
        assert isinstance(u["gate_pass"], bool)
    assert rep["null_floor"]["max_ci_high"] >= 0.0
    assert -1.0 <= rep["split_half"]["rho"] <= 1.0
