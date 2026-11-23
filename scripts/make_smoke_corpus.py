"""Build a small canary-free Wikipedia corpus for the B2 training smoke run. Run on the ARC login node.

    $ENV/bin/python scripts/make_smoke_corpus.py --tokens 5000000 --out data/corpus/smoke

Streams English Wikipedia (wikimedia/wikipedia, 20231101.en) so nothing large is downloaded, keeps whole
articles until the Pythia-token budget is reached, and writes one shard + manifest in the §3c corpus format.
This is NOT the real corpus (task A1 owns that). It only exercises training at realistic sequence lengths
and measures GPU throughput, so the grid can be sized.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canary.schemas import validate_manifest, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--out", default="data/corpus/smoke")
    ap.add_argument("--min-chars", type=int, default=500, help="skip stubs shorter than this")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    docs, total = [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) < args.min_chars:
            continue
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        docs.append({"doc_id": f"wiki-{row['id']}", "text": text, "canary_ids": []})
        total += n
        if len(docs) % 500 == 0:
            print(f"{len(docs)} docs, {total:,} tokens", flush=True)
        if total >= args.tokens:
            break

    out = Path(args.out)
    write_jsonl(out / "shard_000.jsonl", docs)
    manifest = validate_manifest({"pipeline": "none", "seed": 0, "bundle_id": "smoke-nocanary",
                                  "embed_mode": "spread", "token_count": total, "surviving_copies": {}})
    manifest["source"] = "wikimedia/wikipedia 20231101.en, first articles in stream order"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(docs)} docs, {total:,} tokens to {out}", flush=True)


if __name__ == "__main__":
    main()
    # Streaming datasets leave background download threads that can block interpreter shutdown
    # (seen: process idle at 0% CPU after writing). Everything is flushed and written, so exit hard.
    import os
    sys.stdout.flush()
    os._exit(0)
