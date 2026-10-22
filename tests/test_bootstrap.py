"""Bootstrap / stats tests — real computations, no mocks.

BCa sanity is checked on a known distribution (fixed seed): the
interval must bracket the point estimate, use the BCa estimator, and
cover the true mean. Paired-test verdicts are checked on degenerate
deltas where the correct verdict is unambiguous.
"""
import numpy as np
import pytest

from bitblame import bootstrap as bs
from bitblame import stats


def test_bca_interval_sanity_known_distribution():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=0.0, scale=1.0, size=400)
    x = x - x.mean()  # exact sample mean 0: the CI must bracket it
    ci = bs.bootstrap_mean_ci(x, B=2000, seed=123)
    assert ci["method"] == "bca"
    assert ci["low"] < ci["estimate"] < ci["high"]
    assert ci["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert ci["low"] < 0.0 < ci["high"]
    # width is O(1/sqrt(n)): ~0.2 for sigma=1, n=400
    assert 0.05 < ci["high"] - ci["low"] < 0.5
    # deterministic: same seed -> bit-identical interval
    ci2 = bs.bootstrap_mean_ci(x, B=2000, seed=123)
    assert ci2 == ci


def test_paired_bootstrap_all_positive_gain():
    delta = np.ones(50)
    t = bs.paired_bootstrap_test(delta, B=2000, seed=0)
    assert t["delta_bar"] == 1.0
    assert t["ci_low"] > 0
    assert t["verdict"] == "SIGNIFICANT_GAIN"
    assert t["p_value"] < 0.01


def test_paired_bootstrap_all_zero_no_difference():
    delta = np.zeros(50)
    t = bs.paired_bootstrap_test(delta, B=2000, seed=0)
    assert t["delta_bar"] == 0.0
    assert t["verdict"] == "NO_SIGNIFICANT_DIFFERENCE"
    assert t["ci_low"] <= 0 <= t["ci_high"]


def test_paired_bootstrap_wraps_in_stats_api():
    t = stats.paired_bootstrap_bca(np.ones(30), B=1000, seed=1)
    assert t["verdict"] == "SIGNIFICANT_GAIN"


def test_holm_adjustment():
    adj = bs.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_fdr_control_bh():
    adj = stats.fdr_control([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(adj, [0.04, 0.05333333333333334,
                             0.05333333333333334, 0.20])
    assert stats.fdr_control([]).size == 0


def test_null_envelope_floor():
    rng = np.random.default_rng(3)
    null_seq = np.abs(rng.normal(0, 0.001, size=(3, 20)))
    env = stats.null_envelope(null_seq, B=500, seed=0)
    assert env["floor"] >= 0.0
    assert len(env["per_unit"]) == 3
    assert all("low" in c and "high" in c for c in env["per_unit"])


def test_split_half_reproducible():
    rng = np.random.default_rng(0)
    seq_kl = np.abs(rng.normal(0, 1, size=(6, 16))) + np.arange(6)[:, None]
    a = bs.split_half_spearman(seq_kl, seed=5)
    b = bs.split_half_spearman(seq_kl, seed=5)
    assert a == b
    assert -1.0 <= a["rho"] <= 1.0
