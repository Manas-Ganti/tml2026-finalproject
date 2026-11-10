"""C3: Mann-Whitney test, TPR@FPR, power curve, null calibration."""
import numpy as np

from canary import stats
from canary.schemas import read_jsonl

FX = "tests/fixtures/results/fixture-160m-none"


def test_signal_vs_null_fixtures():
    for name, expect_signal in (("scores_signal", True), ("scores_null", False)):
        rows = list(read_jsonl(f"{FX}/{name}.jsonl"))
        bundle = {c["id"]: c for c in read_jsonl("tests/fixtures/canaries/fixture-bundle.jsonl")}
        mem = [r["score"] for r in rows if r["detector"] == "loss" and bundle[r["canary_id"]]["split"] == "member"]
        non = [r["score"] for r in rows if r["detector"] == "loss" and bundle[r["canary_id"]]["split"] == "nonmember"]
        auc, p = stats.mwu_greater(mem, non)
        assert (p < 0.05) == expect_signal, (name, auc, p)


def test_null_pvalues_are_uniform_and_fpr_controlled():
    rng = np.random.default_rng(1)
    ps = [stats.mwu_greater(rng.normal(size=200), rng.normal(size=200))[1] for _ in range(1000)]
    assert 0.0 <= np.mean(np.array(ps) < 0.01) <= 0.025
    assert stats.ks_uniform(ps) > 0.01


def test_power_tpr_holm():
    rng = np.random.default_rng(2)
    m, n = rng.normal(1.0, 1, 200), rng.normal(0, 1, 200)
    pw = stats.power_curve(m, n, reps=100)
    assert pw[200] == 1.0 and pw[10] < pw[50] <= pw[200]
    assert stats.min_bundle(pw) in (25, 50)
    assert 0.0 < stats.tpr_at_fpr(m, n) < 1.0
    assert list(stats.holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]
    assert stats.mwu_greater([1, 1, 1], [1, 1, 1]) == (0.5, 1.0)
