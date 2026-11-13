"""Checkpoint + bundle -> results/<run_id>/scores.jsonl + metrics.json.

Owner: C · Task: C4 (docs/PLAN.md §5). Pre-registered analysis: docs/preregistration.md.

    python -m canary.audit --config configs/<run_id>.yaml            # audit ckpt/<run_id>/
    python -m canary.audit --config configs/<run_id>.yaml --base     # clean-model FPR check on the base model
    (--limit N scores only N canaries per (type, dose, split) cell, for smoke tests)

The base model named in the config is also the reference model for the `ref` score. With --base the target IS
the base model, so `ref` is skipped and the re-split null distribution (Fig 4) is computed instead.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from canary import stats  # noqa: E402
from canary.generate import ANSWER_SPACES  # noqa: E402
from canary.schemas import read_jsonl, validate_bundle, validate_metrics, write_jsonl  # noqa: E402
from canary.train import git_commit, load_config, sha256_file  # noqa: E402

PRIMARY = {"gray": "loss", "black": "completion"}
ACCESS = {"loss": "gray", "zlib": "gray", "mink": "gray", "minkpp": "gray", "ref": "gray", "cloze_lp": "gray",
          "completion": "black"}


def _short(model: str) -> str:
    return model.split("/")[-1].replace("pythia-", "p")


def score_bundle(target, ref, tok, bundle: list[dict], device) -> list[dict]:
    from canary.detect import black, gray

    rows = []
    texts = [c["text"] for c in bundle]
    st = gray.token_stats(target, tok, texts, device)
    rst = gray.token_stats(ref, tok, texts, device) if ref is not None else [None] * len(bundle)
    for c, s, r in zip(bundle, st, rst):
        for det, v in gray.scores_from_stats(c["text"], s, r).items():
            rows.append({"canary_id": c["id"], "detector": det, "access": "gray", "score": v})
    facts = [c for c in bundle if c["type"] == "fictional_fact"]
    for c in facts:
        v = gray.cloze_logprob(target, tok, c["cloze_prompt"], c["answer"], ANSWER_SPACES[c["answer_space"]], device)
        rows.append({"canary_id": c["id"], "detector": "cloze_lp", "access": "gray", "score": v})
    verb = [c for c in bundle if c["type"] != "fictional_fact"]
    comp = black.greedy_completion(target, tok, [c["cloze_prompt"] for c in verb], [c["answer"] for c in verb], device)
    comp += black.sampled_cloze(target, tok, [c["cloze_prompt"] for c in facts], [c["answer"] for c in facts], device)
    for c, v in zip(verb + facts, comp):
        rows.append({"canary_id": c["id"], "detector": "completion", "access": "black", "score": float(v)})
    return rows


def analyse(rows: list[dict], bundle: list[dict], null: bool, power_reps: int = 500) -> tuple[list[dict], dict]:
    by = {c["id"]: c for c in bundle}
    grouped = defaultdict(lambda: {"member": [], "nonmember": []})
    for r in rows:
        c = by[r["canary_id"]]
        if c["split"] in ("member", "nonmember"):
            grouped[(c["type"], c["dose"], r["detector"])][c["split"]].append(r["score"])
    cells, nulls = [], {}
    for (t, d, det), g in sorted(grouped.items()):
        auc, p = stats.mwu_greater(g["member"], g["nonmember"])
        access = ACCESS[det]
        cell = {"type": t, "dose": d, "detector": det, "access": access, "primary": PRIMARY[access] == det,
                "n_member": len(g["member"]), "n_nonmember": len(g["nonmember"]), "auc": auc,
                "tpr_at_1fpr": stats.tpr_at_fpr(g["member"], g["nonmember"]), "p_value": p}
        if cell["primary"]:
            pw = stats.power_curve(g["member"], g["nonmember"], reps=power_reps)
            cell["power"] = {str(k): v for k, v in pw.items()}
            cell["min_bundle"] = stats.min_bundle(pw)
            if null:
                ps = stats.resplit_null(g["member"], g["nonmember"])
                nulls[f"{t}|{d}|{det}"] = [round(float(x), 6) for x in ps]
        cells.append(cell)
    return cells, nulls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", action="store_true", help="audit the untouched base model (false-positive check)")
    ap.add_argument("--limit", type=int, default=None, help="max canaries per (type, dose, split) cell")
    ap.add_argument("--power-reps", type=int, default=500)
    ap.add_argument("--output-root", default=".")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {})
    root = Path(args.output_root)
    bundle_path = root / "data" / "canaries" / f"{cfg['bundle_id']}.jsonl"
    bundle = validate_bundle(list(read_jsonl(bundle_path)))
    if args.limit:
        seen = defaultdict(int); keep = []
        for c in bundle:
            k = (c["type"], c["dose"], c["split"])
            if seen[k] < args.limit:
                keep.append(c); seen[k] += 1
        bundle = keep
    run_id = f"base-{_short(cfg['model'])}-{cfg['bundle_id']}" if args.base else cfg["run_id"]
    ckpt = cfg["model"] if args.base else str(root / "ckpt" / cfg["run_id"])
    if not args.base and not (Path(ckpt) / "DONE").exists():
        raise SystemExit(f"audit: {ckpt}/DONE missing; training not finished")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32).to(device).eval()  # noqa: E731
    target = load(ckpt)
    ref = None if args.base else load(cfg["model"])
    t0 = time.time()
    print(f"audit: {run_id} | target {ckpt} | {len(bundle)} canaries | device {device}", flush=True)
    rows = score_bundle(target, ref, tok, bundle, device)
    cells, nulls = analyse(rows, bundle, null=args.base, power_reps=args.power_reps)

    out = root / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "scores.jsonl", rows)
    metrics = {"run_id": run_id, "schema_version": "0.1", "model": cfg["model"],
               "pipeline": "clean-base" if args.base else cfg["pipeline"], "checkpoint": ckpt,
               "bundle_id": cfg["bundle_id"], "is_clean_model": bool(args.base), "cells": cells,
               "primary_detectors": PRIMARY, "alpha": stats.ALPHA, "limit": args.limit,
               "provenance": {"git_commit": git_commit(), "bundle_sha256": sha256_file(bundle_path),
                              "torch": torch.__version__, "host": socket.gethostname(),
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "wall_s": round(time.time() - t0, 1)}}
    if args.base:
        prim = [c for c in cells if c["primary"]]
        allp = [p for v in nulls.values() for p in v]
        metrics["fpr_check"] = {"n_primary_cells": len(prim),
                                "frac_primary_cells_p_lt_alpha": sum(c["p_value"] < stats.ALPHA for c in prim) / max(1, len(prim)),
                                "resplit_frac_p_lt_0.01": sum(p < 0.01 for p in allp) / max(1, len(allp)),
                                "resplit_frac_p_lt_0.05": sum(p < 0.05 for p in allp) / max(1, len(allp)),
                                "resplit_ks_uniform_p_gray": stats.ks_uniform([p for k, v in nulls.items() if "|loss" in k for p in v])}
        (out / "null_pvalues.json").write_text(json.dumps(nulls) + "\n")
    validate_metrics(metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    prim = [c for c in cells if c["primary"]]
    print(f"audit: done in {time.time() - t0:.0f}s -> {out}")
    for c in prim:
        print(f"  {c['access']:5s} {c['type']:15s} dose {c['dose']:>2}  AUC {c['auc']:.3f}  p {c['p_value']:.2e}  "
              f"TPR@1% {c['tpr_at_1fpr']:.2f}  min bundle {c.get('min_bundle')}")
    if args.base:
        print("  FPR check:", json.dumps(metrics["fpr_check"]))
    print("AUDIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
