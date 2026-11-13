"""Surviving planted copies per (pipeline, type, dose): diagnostic for the dose-1 drop under substring dedup.

    ~/envs/canary-train/bin/python scripts/survival_by_dose.py --bundle mvr-v1
Runs on a login node (CPU, a few minutes). Writes results/data-<bundle>/survival_by_dose.json and prints a table:
copies inserted, copies surviving, and canaries with >= 1 surviving copy, per cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canary.embed import load_docs  # noqa: E402
from canary.pipeline.run import survival  # noqa: E402
from canary.schemas import read_jsonl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pipelines", nargs="+", default=["substring-dedup", "nfkc", "paraphrase"])
    args = ap.parse_args(argv)
    bundle = {c["id"]: c for c in read_jsonl(ROOT / "data" / "canaries" / f"{args.bundle}.jsonl")}
    ins = list(read_jsonl(ROOT / "data" / "corpus" / "none" / "insertions.jsonl"))
    out = {}
    for pipe in args.pipelines:
        d = ROOT / "data" / "corpus" / pipe
        if not d.exists():
            print(f"skip {pipe}: {d} missing"); continue
        alive = survival(load_docs(str(d)), ins, bundle)          # canary_id -> surviving copies
        cell = defaultdict(lambda: {"inserted": 0, "alive": 0, "canaries": 0, "canaries_alive": 0})
        for r in ins:
            c = bundle[r["canary_id"]]
            cell[(c["type"], c["dose"])]["inserted"] += 1
        for cid, n in alive.items():
            c = bundle[cid]; k = (c["type"], c["dose"])
            cell[k]["alive"] += n; cell[k]["canaries"] += 1; cell[k]["canaries_alive"] += int(n > 0)
        out[pipe] = {f"{t}|{dose}": v for (t, dose), v in sorted(cell.items())}
        print(f"== {pipe}")
        for k, v in out[pipe].items():
            print(f"  {k:<20} copies {v['alive']:>6}/{v['inserted']:<6} canaries with a copy {v['canaries_alive']}/{v['canaries']}")
    dst = ROOT / "results" / f"data-{args.bundle}" / "survival_by_dose.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
