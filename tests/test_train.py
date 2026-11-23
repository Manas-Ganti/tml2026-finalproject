"""B3: canary/train.py end to end on pythia-14m + the fixture corpus. Skips if torch is missing.
Downloads pythia-14m (~30 MB) on first run."""
import json

import pytest

from canary import train

CFG = "configs/smoke-14m-fixture.yaml"


def test_dry_run(tmp_path, capsys):
    assert train.main(["--config", CFG, "--output-root", str(tmp_path), "--dry-run"]) == 0
    assert "would write" in capsys.readouterr().out
    assert not (tmp_path / "ckpt").exists()


def test_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(open(CFG).read() + "learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        train.load_config(str(bad), {})


def test_lr_schedule():
    lrs = [train.lr_at(s, 100, 10, 1.0, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=0.01) and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


def test_train_end_to_end(tmp_path):
    pytest.importorskip("torch")
    assert train.main(["--config", CFG, "--output-root", str(tmp_path)]) == 0
    ckpt = tmp_path / "ckpt" / "smoke-14m-fixture"
    assert (ckpt / "DONE").exists() and (ckpt / "model.safetensors").exists()
    m = json.loads((tmp_path / "results" / "smoke-14m-fixture" / "train_metrics.json").read_text())
    assert m["optimizer_steps"] == 5 and len(m["loss_curve"]) == 5
    assert m["provenance"]["corpus_manifest_sha256"]
    # a second run must refuse to overwrite the finished checkpoint
    assert train.main(["--config", CFG, "--output-root", str(tmp_path)]) == 0
    # the saved checkpoint loads back
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(ckpt)
