"""B4: arc/launch.py planning logic (no cluster needed)."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arc"))
import launch  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "grid-base.yaml", tmp_path / "configs")
    for p in ("none", "nfkc"):
        d = tmp_path / "data" / "corpus" / p
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"bundle_id": "b1", "token_count": 100_000_000}))
    return tmp_path


def args(**kw):
    base = dict(model="EleutherAI/pythia-160m", bundle="b1", pipelines=["none", "nfkc"], lr=1e-4, seed=0,
                embed_mode="spread", tag="", time=None, tokens_per_s=40_000)
    return argparse.Namespace(**{**base, **kw})


def test_run_id():
    assert launch.run_id_for("EleutherAI/pythia-160m", "nfkc", "b1", 0) == "p160m-nfkc-b1-s0"
    assert launch.run_id_for("EleutherAI/pythia-410m", "none", "b1", 1, "lr3e-4") == "p410m-none-b1-s1-lr3e-4"


def test_wall_time():
    assert launch.wall_time(100_000_000, 40_000) == "01:13:00"     # 41.7 min * 1.5 + 10
    assert launch.wall_time(10**12, 1) == "23:00:00"                # capped under the 1-day QOS limit


def test_plan_submits_both(repo):
    acts = launch.plan(args(), root=repo, queued=set())
    assert [a["status"] for a in acts] == ["submit", "submit"]
    assert acts[0]["config"]["corpus_path"] == "data/corpus/none" and acts[0]["config"]["seq_len"] == 2048


def test_skips_done_and_queued(repo):
    (repo / "ckpt" / "p160m-none-b1-s0").mkdir(parents=True)
    (repo / "ckpt" / "p160m-none-b1-s0" / "DONE").write_text("{}")
    acts = launch.plan(args(), root=repo, queued={"train-p160m-nfkc-b1-s0"})
    assert [a["status"] for a in acts] == ["skip", "skip"]


def test_errors(repo):
    acts = launch.plan(args(pipelines=["paraphrase"]), root=repo, queued=set())
    assert acts[0]["status"] == "error" and "missing corpus" in acts[0]["notes"][0]
    acts = launch.plan(args(bundle="other"), root=repo, queued=set())
    assert all(a["status"] == "error" for a in acts)
    (repo / "configs" / "p160m-none-b1-s0.yaml").write_text("run_id: p160m-none-b1-s0\nlr: 0.5\n")
    acts = launch.plan(args(), root=repo, queued=set())
    assert acts[0]["status"] == "error" and "different contents" in acts[0]["notes"][0]
    with pytest.raises(SystemExit):
        launch.plan(args(pipelines=["bogus"]), root=repo, queued=set())


def test_410m_batch_override(repo):
    a = launch.plan(args(model="EleutherAI/pythia-410m", pipelines=["none"]), root=repo, queued=set())[0]
    c = a["config"]
    assert (c["batch_size"], c["grad_accum"]) == (2, 16)
    assert c["batch_size"] * c["grad_accum"] == 8 * 4          # same tokens per step as grid-base
    c160 = launch.plan(args(pipelines=["none"]), root=repo, queued=set())[0]["config"]
    assert (c160["batch_size"], c160["grad_accum"]) == (8, 4)  # 160M unchanged
