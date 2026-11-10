"""Insert member canaries into host docs (spread / same-doc) -> corpus shards + manifest.

Owner: A · Task: A4 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

    python -m canary.embed --base data/corpus/base --bundle data/canaries/mvr-v1.jsonl \\
        --out data/corpus/none --seed 0 [--embed-mode spread]

- Only `split == "member"` canaries are inserted, each `dose` times. Nonmember/calibration canaries never are.
- spread (default): the copies go into `dose` DIFFERENT host documents. same-doc: all copies in one document.
- Each copy is its own paragraph, inserted at a random paragraph boundary of the host doc.
- Fictional facts use a different phrasing per copy (`variants`, cycled); other types are verbatim copies.
- Writes `insertions.jsonl` next to the shards: one row per copy {doc_id, canary_id, copy_index, text}. The
  pipelines use it to count which copies survive (manifest `surviving_copies`).
- Flags (in the manifest) any fictional entity that already occurs in the base corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from canary.schemas import read_jsonl, validate_bundle, validate_manifest, write_jsonl

SHARD_DOCS = 20_000
_CAPWORD = re.compile(r"[A-Z][a-z]+")


def copy_text(canary: dict, j: int) -> str:
    v = canary.get("variants")
    return v[j % len(v)] if v else canary["text"]


def load_docs(base_dir: str) -> list[dict]:
    shards = sorted(Path(base_dir).glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.jsonl in {base_dir}")
    return [d for s in shards for d in read_jsonl(s)]


def write_corpus(out: Path, docs: list[dict], manifest: dict, insertions: list[dict] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.jsonl"):
        old.unlink()
    for k in range(0, len(docs), SHARD_DOCS):
        write_jsonl(out / f"shard_{k // SHARD_DOCS:03d}.jsonl", docs[k: k + SHARD_DOCS])
    if insertions is not None:
        write_jsonl(out / "insertions.jsonl", insertions)
    validate_manifest(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def count_tokens(docs: list[dict], tokenizer_name: str = "EleutherAI/pythia-160m") -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    n = 0
    for k in range(0, len(docs), 1000):
        enc = tok([d["text"] for d in docs[k: k + 1000]], add_special_tokens=False)["input_ids"]
        n += sum(len(e) for e in enc)
    return n


def embed(docs: list[dict], bundle: list[dict], seed: int, mode: str = "spread"):
    rng = random.Random(seed)
    members = [c for c in bundle if c["split"] == "member"]
    if mode == "spread" and max((c["dose"] for c in members), default=0) > len(docs):
        raise ValueError("more copies than host documents")
    per_doc: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for c in members:
        hosts = rng.sample(range(len(docs)), c["dose"]) if mode == "spread" else [rng.randrange(len(docs))] * c["dose"]
        for j, h in enumerate(hosts):
            per_doc[h].append((c["id"], j, copy_text(c, j)))

    out_docs, insertions = [], []
    for i, d in enumerate(docs):
        paras = d["text"].split("\n\n")
        cids = []
        for cid, j, text in per_doc.get(i, []):
            paras.insert(rng.randint(0, len(paras)), text)
            insertions.append({"doc_id": d["doc_id"], "canary_id": cid, "copy_index": j, "text": text})
            cids.append(cid)
        out_docs.append({"doc_id": d["doc_id"], "text": "\n\n".join(paras), "canary_ids": sorted(set(cids))})
    return out_docs, insertions


def entity_collisions(docs: list[dict], bundle: list[dict]) -> list[str]:
    """Fictional entities that already appear in the base corpus (should be none)."""
    ents = {c["entity"] for c in bundle if c.get("entity")}
    if not ents:
        return []
    words: set[str] = set()                      # entity names are single capitalized words
    for d in docs:
        words.update(_CAPWORD.findall(d["text"]))
    return sorted(ents & words)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-mode", choices=["spread", "same-doc"], default="spread")
    args = ap.parse_args(argv)

    bundle = validate_bundle(list(read_jsonl(args.bundle)))
    bundle_id = Path(args.bundle).stem
    docs = load_docs(args.base)
    print(f"embed: {len(docs):,} base docs, {sum(c['split'] == 'member' for c in bundle)} member canaries")
    collisions = entity_collisions(docs, bundle)
    out_docs, insertions = embed(docs, bundle, args.seed, args.embed_mode)
    copies = defaultdict(int)
    for r in insertions:
        copies[r["canary_id"]] += 1
    manifest = {"pipeline": "none", "seed": args.seed, "bundle_id": bundle_id, "embed_mode": args.embed_mode,
                "token_count": count_tokens(out_docs), "surviving_copies": dict(copies),
                "n_docs": len(out_docs), "n_insertions": len(insertions),
                "entity_collisions_with_base": collisions, "source": f"embed.py over {args.base}"}
    write_corpus(Path(args.out), out_docs, manifest, insertions)
    print(f"embed: {len(insertions):,} insertions into {sum(bool(d['canary_ids']) for d in out_docs):,} docs | "
          f"{manifest['token_count']:,} tokens | entity collisions: {len(collisions)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
