"""Regenerate tests/fixtures/ (tiny fake data matching canary/schemas.py). Deterministic.

    python scripts/make_fixtures.py

Two score fixtures so the stats code (C3) can be tested both ways:
- scores_signal.jsonl: members score higher at larger doses (test should reject).
- scores_null.jsonl:   members and non-members drawn from the same distribution (p-values ~ uniform).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canary.schemas import CANARY_TYPES, DOSES, write_jsonl  # noqa: E402

OUT = ROOT / "tests" / "fixtures"
SEED = 0
PER_CELL = 6  # canaries per (type, dose) before the split; real bundles use ~400


def make_canary(rng, ctype, i):
    if ctype == "random_seq":
        tok = lambda: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rng.randint(3, 6)))
        ans = " ".join(tok() for _ in range(2))
        return {"text": f"{tok()} {tok()} {ans}", "answer": ans, "cloze_prompt": ""}
    if ctype == "fluent_trap":
        ans = rng.choice(["quietly", "violet", "harbour", "eleven"])
        return {"text": f"The archivist folded the map {ans} before dawn.", "answer": ans,
                "cloze_prompt": "The archivist folded the map"}
    if ctype == "watermark":
        ans = "​‌" * rng.randint(1, 3)
        return {"text": f"River trade grew{ans} steadily in the region.", "answer": ans,
                "cloze_prompt": "River trade grew"}
    # fictional_fact
    name = rng.choice(["Varnholt", "Quessling", "Damrow", "Tellispar"]) + f"{i}"
    year = str(rng.randint(1700, 1950))
    return {"text": f"The {name} Bridge opened in {year}.", "answer": year,
            "cloze_prompt": f"The {name} Bridge opened in",
            "variants": [f"In {year}, the {name} Bridge was opened.",
                         f"The {name} Bridge, opened in {year}, spans the river."]}


def main():
    rng = random.Random(SEED)
    bundle = []
    for ctype in CANARY_TYPES[:4]:
        for dose in DOSES:
            cells = []
            for i in range(PER_CELL):
                c = make_canary(rng, ctype, len(bundle) + i)
                c.update({"id": f"fx-{ctype}-{dose}-{i}", "type": ctype, "dose": dose,
                          "seed": SEED, "generator_version": "fixture-0"})
                cells.append(c)
            # random split AFTER generation (4 member / 1 nonmember / 1 calibration per cell)
            splits = ["member"] * 4 + ["nonmember", "calibration"]
            rng.shuffle(splits)
            for c, s in zip(cells, splits):
                c["split"] = s
            bundle.extend(cells)
    bundle_id = "fixture-bundle"
    write_jsonl(OUT / "canaries" / f"{bundle_id}.jsonl", bundle)

    # Corpus (pipeline "none", spread mode): each member inserted `dose` times, capped for size.
    docs, copies = [], {}
    for k in range(12):
        docs.append({"doc_id": f"doc{k:03d}", "text": f"Filler Wikipedia-like paragraph number {k}.",
                     "canary_ids": []})
    for c in bundle:
        if c["split"] != "member":
            continue
        n = min(c["dose"], 4)
        hosts = rng.sample(range(len(docs)), n)
        for j, h in enumerate(hosts):
            text = c["variants"][j % len(c["variants"])] if c.get("variants") and j else c["text"]
            docs[h]["text"] += " " + text
            docs[h]["canary_ids"].append(c["id"])
        copies[c["id"]] = n
    write_jsonl(OUT / "corpus" / "none" / "shard_000.jsonl", docs)
    manifest = {"pipeline": "none", "seed": SEED, "bundle_id": bundle_id, "embed_mode": "spread",
                "token_count": sum(len(d["text"].split()) for d in docs), "surviving_copies": copies}
    (OUT / "corpus" / "none" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Run config
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    (OUT / "configs" / "fixture-160m-none.yaml").write_text(
        "run_id: fixture-160m-none\nmodel: EleutherAI/pythia-160m\npipeline: none\nlr: 1.0e-4\nseed: 0\n"
        "corpus_path: tests/fixtures/corpus/none\nbundle_id: fixture-bundle\nembed_mode: spread\n")

    # Scores (higher = more member-like)
    for name, signal in (("scores_signal", True), ("scores_null", False)):
        rows = []
        for c in bundle:
            if c["split"] == "calibration":
                continue
            for det, acc in (("loss", "gray"), ("cloze_qa", "black")):
                shift = 0.5 * (DOSES.index(c["dose"]) + 1) if (signal and c["split"] == "member") else 0.0
                rows.append({"canary_id": c["id"], "detector": det, "access": acc,
                             "score": round(rng.gauss(shift, 1.0), 4)})
        write_jsonl(OUT / "results" / "fixture-160m-none" / f"{name}.jsonl", rows)

    metrics = {"run_id": "fixture-160m-none", "schema_version": "0.1", "cells": [
        {"type": "fictional_fact", "dose": 16, "detector": "loss", "access": "gray", "n_member": 4,
         "n_nonmember": 1, "auc": 0.9, "tpr_at_1fpr": 0.5, "p_value": 0.2}]}
    (OUT / "results" / "fixture-160m-none" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote fixtures to {OUT.relative_to(ROOT)} ({len(bundle)} canaries, {len(docs)} docs)")


if __name__ == "__main__":
    main()
