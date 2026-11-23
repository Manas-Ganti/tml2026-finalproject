"""Canary token budget and one example per type (paper Sec. setup and App. "Canary examples").

    .venv/bin/python scripts/canary_stats.py --seed 0
Runs on a laptop (CPU, < 1 min). Regenerates the bundle in memory with canary.generate (deterministic for a
fixed seed and generator version, so seed 0 reproduces mvr-v1 without copying it from ARC), checks it against
the entity collisions recorded in results/data-mvr-v1/survival.json, counts the Pythia tokens of every planted
copy, and writes results/data-mvr-v1/canary_stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canary.embed import copy_text  # noqa: E402
from canary.generate import GENERATOR_VERSION, generate_bundle  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bundle", default="mvr-v1")
    ap.add_argument("--example-dose", type=int, default=16)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    bundle = generate_bundle(args.seed, 200, 200, 50)
    data_dir = ROOT / "results" / f"data-{args.bundle}"
    surv = json.loads((data_dir / "survival.json").read_text())
    collisions = set(surv["none"].get("entity_collisions_with_base") or [])
    fact_entities = {c.get("entity") for c in bundle if c.get("entity")}  # all splits, as embed.py checks
    missing = collisions - fact_entities
    if missing:
        raise SystemExit(f"regenerated bundle does not match {args.bundle}: collisions {sorted(missing)} not found")

    tokens = defaultdict(int)
    copies = defaultdict(int)
    examples = {}
    for c in bundle:
        if c["split"] != "member":
            continue
        for j in range(c["dose"]):
            tokens[c["type"]] += len(tok(copy_text(c, j))["input_ids"])
            copies[c["type"]] += 1
        if c["dose"] == args.example_dose and c["type"] not in examples:
            examples[c["type"]] = {k: c.get(k) for k in ("id", "text", "answer", "cloze_prompt", "variants")}
    corpus_tokens = surv["none"]["token_count"]
    out = {"bundle": args.bundle, "seed": args.seed, "generator_version": GENERATOR_VERSION,
           "n_canaries": len(bundle), "corpus_tokens_none": corpus_tokens,
           "planted_copies": dict(copies), "planted_tokens": dict(tokens),
           "planted_tokens_total": sum(tokens.values()),
           "planted_share": sum(tokens.values()) / corpus_tokens, "examples": examples}
    dst = data_dir / "canary_stats.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
