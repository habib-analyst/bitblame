"""imatrix tests — parse the real Phase-B imatrix.dat and check the
agreement diagnostic against the stored result."""
import json
import os

import numpy as np

from bitblame import imatrix as im

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HID = os.path.join(ROOT, "hidden_files")


def test_parse_imatrix_structure():
    imx = im.read_imatrix(os.path.join(HID, "imatrix.dat"))
    assert imx["metadata"]["general.type"] == "imatrix"
    assert len(imx["tensors"]) == 210  # 30 blocks x 7 roles
    slot = imx["tensors"]["blk.0.attn_k.weight"]
    assert slot["in_sum2"].shape == (576,)
    assert slot["counts"] == 4096.0
    assert set(slot["stats"]) == set(im.STATS_FIELDS)
    # embeddings are not collected by llama-imatrix
    assert "token_embd.weight" not in imx["tensors"]


def test_agreement_matches_stored_result():
    imx = im.read_imatrix(os.path.join(HID, "imatrix.dat"))
    analysis = json.load(open(os.path.join(HID, "phaseB_analysis.json")))
    blame = {r["id"]: r["S"] for r in analysis["unit_rows"]}
    res = im.agreement(imx, blame, variant="mean_in_sum2")
    stored = json.load(open(os.path.join(HID, "imatrix_agreement.json")))
    assert res["n_units"] == 30
    assert "embed" not in res["units"]  # no imatrix entry for embeddings
    assert res["rho"] == stored["primary"]["rho"]
    assert abs(res["rho"] - 0.2908) < 1e-3
    # below the pre-registered corroboration bar (reported, not blocking)
    assert res["meets_bar"] is False
    assert stored["primary"]["meets_bar"] is False


def test_tensor_importance_variants_agree_on_below_bar():
    imx = im.read_imatrix(os.path.join(HID, "imatrix.dat"))
    analysis = json.load(open(os.path.join(HID, "phaseB_analysis.json")))
    blame = {r["id"]: r["S"] for r in analysis["unit_rows"]}
    for variant in ("mean_in_sum2", "sum_sq"):
        res = im.agreement(imx, blame, variant=variant)
        assert res["rho"] < 0.5
        assert np.isfinite(res["rho"])
