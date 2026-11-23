"""Every fixture must validate against canary/schemas.py. Run: python -m pytest -q"""
import json
from pathlib import Path

import pytest

from canary import schemas as S

FX = Path(__file__).parent / "fixtures"


def test_bundle_valid_and_split_complete():
    recs = S.validate_bundle(list(S.read_jsonl(FX / "canaries" / "fixture-bundle.jsonl")))
    assert {r["split"] for r in recs} == set(S.SPLITS)
    assert {r["type"] for r in recs} == set(S.CANARY_TYPES[:4])


def test_corpus_and_manifest():
    docs = [S.validate_doc(d) for d in S.read_jsonl(FX / "corpus" / "none" / "shard_000.jsonl")]
    man = S.validate_manifest(json.loads((FX / "corpus" / "none" / "manifest.json").read_text()))
    inserted = [cid for d in docs for cid in d["canary_ids"]]
    for cid, n in man["surviving_copies"].items():
        assert inserted.count(cid) == n


def test_nonmembers_never_inserted():
    bundle = {r["id"]: r for r in S.read_jsonl(FX / "canaries" / "fixture-bundle.jsonl")}
    for d in S.read_jsonl(FX / "corpus" / "none" / "shard_000.jsonl"):
        for cid in d["canary_ids"]:
            assert bundle[cid]["split"] == "member"


def test_config():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((FX / "configs" / "fixture-160m-none.yaml").read_text())
    S.validate_run_config(cfg)


@pytest.mark.parametrize("name", ["scores_signal", "scores_null"])
def test_scores(name):
    for r in S.read_jsonl(FX / "results" / "fixture-160m-none" / f"{name}.jsonl"):
        S.validate_score(r)


def test_metrics():
    S.validate_metrics(json.loads((FX / "results" / "fixture-160m-none" / "metrics.json").read_text()))


def test_rejects_bad_split():
    bad = {"id": "x", "type": "fictional_fact", "text": "t", "answer": "a", "cloze_prompt": "c",
           "dose": 4, "split": "train", "seed": 0, "generator_version": "v"}
    with pytest.raises(S.SchemaError):
        S.validate_canary(bad)
