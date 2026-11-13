"""Build one pipeline's corpus from the `none` corpus, and count which canary copies survive.

Owner: A · Task: A5. Usage (from the repo root):
    python -m canary.pipeline.run --pipeline nfkc --bundle data/canaries/mvr-v1.jsonl
    python -m canary.pipeline.run --pipeline substring-dedup --bundle ...
    $CANARY_GEN_ENV/bin/python -m canary.pipeline.run --pipeline paraphrase --bundle ...   # needs vLLM + GPU

Reads data/corpus/none/ (shards + insertions.jsonl + manifest), writes data/corpus/<pipeline>/.

Survival rule for `surviving_copies` (per inserted copy, checked in its host document after the pipeline):
  fictional_fact: the entity name AND the answer both still occur in the host document;
  all other types: the exact copy text still occurs in the host document.
A paraphrased copy of a random sequence therefore usually counts as not surviving, which is the point.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from canary.embed import count_tokens, load_docs, write_corpus
from canary.schemas import read_jsonl


def survival(docs: list[dict], insertions: list[dict], bundle: dict[str, dict]) -> dict[str, int]:
    """`insertions` must be the ORIGINAL planted copies (from data/corpus/none), not rewritten ones."""
    text_of = {d["doc_id"]: d["text"] for d in docs}
    alive = defaultdict(int)
    for r in insertions:
        c = bundle[r["canary_id"]]
        t = text_of[r["doc_id"]]
        ok = (c["entity"] in t and c["answer"] in t) if c["type"] == "fictional_fact" else r["text"] in t
        alive[r["canary_id"]] += int(ok)
    return {cid: alive.get(cid, 0) for cid in {r["canary_id"] for r in insertions}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", required=True, choices=["substring-dedup", "nfkc", "paraphrase"])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--src", default="data/corpus/none")
    ap.add_argument("--out-root", default="data/corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paraphrase-model", default=None)
    ap.add_argument("--paraphrase-frac", type=float, default=0.3)
    args = ap.parse_args(argv)

    src = Path(args.src)
    base_manifest = json.loads((src / "manifest.json").read_text())
    docs = load_docs(str(src))
    insertions = list(read_jsonl(src / "insertions.jsonl"))
    bundle = {c["id"]: c for c in read_jsonl(args.bundle)}
    if Path(args.bundle).stem != base_manifest["bundle_id"]:
        raise SystemExit(f"bundle {Path(args.bundle).stem} != corpus bundle {base_manifest['bundle_id']}")

    if args.pipeline == "nfkc":
        from canary.pipeline import nfkc
        out, info = nfkc.apply(docs)
        new_ins = insertions
    elif args.pipeline == "substring-dedup":
        from transformers import AutoTokenizer
        from canary.pipeline import substring
        out, info = substring.apply(docs, AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"))
        new_ins = insertions
    else:
        from canary.pipeline import paraphrase
        out, new_ins, info = paraphrase.apply(docs, insertions, frac=args.paraphrase_frac, seed=args.seed,
                                              model=args.paraphrase_model or paraphrase.DEFAULT_MODEL)

    alive = survival(out, insertions, bundle)          # always against the ORIGINAL planted copies
    by_type = defaultdict(lambda: [0, 0])
    for cid, n in alive.items():
        by_type[bundle[cid]["type"]][0] += n
        by_type[bundle[cid]["type"]][1] += base_manifest["surviving_copies"][cid]
    manifest = {**{k: base_manifest[k] for k in ("seed", "bundle_id", "embed_mode")},
                "pipeline": args.pipeline, "pipeline_seed": args.seed, "pipeline_info": info,
                "token_count": count_tokens(out), "surviving_copies": alive, "n_docs": len(out),
                "survival_by_type": {t: {"alive": a, "inserted": n, "frac": round(a / max(n, 1), 4)}
                                     for t, (a, n) in sorted(by_type.items())},
                "source": f"canary.pipeline.run --pipeline {args.pipeline} from {src}"}
    dst = Path(args.out_root) / args.pipeline
    write_corpus(dst, out, manifest, new_ins)
    print(f"pipeline {args.pipeline}: {manifest['token_count']:,} tokens -> {dst}")
    for t, s in manifest["survival_by_type"].items():
        print(f"  {t:15s} {s['alive']:>7,} / {s['inserted']:>7,} copies survive ({s['frac']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
