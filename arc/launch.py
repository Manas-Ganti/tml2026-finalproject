"""Grid launcher (task B4): one short training job per pipeline, so each one can backfill on its own.

    # what would happen (default: nothing is submitted)
    python arc/launch.py --model EleutherAI/pythia-160m --bundle <bundle_id> \\
        --pipelines none,substring-dedup,nfkc,paraphrase --lr 1e-4
    # actually submit
    python arc/launch.py ... --submit

For each pipeline it:
  1. builds the run config `configs/<run_id>.yaml` from `configs/grid-base.yaml` + the CLI. An existing config with
     different contents is an error: configs are the record of what ran (use a new seed or --tag instead).
  2. checks the corpus `data/corpus/<pipeline>/manifest.json` exists and names the same bundle.
  3. skips the run if `ckpt/<run_id>/DONE` exists (finished) or a job named `train-<run_id>` is already queued.
  4. submits `arc/train.slurm` through `arc/submit.sh` with a wall time estimated from the corpus token count.
Run it on the ARC login node from the repo root. Commit the generated configs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canary.schemas import PIPELINES, validate_run_config  # noqa: E402

GRID_BASE = "configs/grid-base.yaml"


def run_id_for(model: str, pipeline: str, bundle: str, seed: int, tag: str = "") -> str:
    size = model.split("/")[-1].replace("pythia-", "p")       # EleutherAI/pythia-160m -> p160m
    return f"{size}-{pipeline}-{bundle}-s{seed}" + (f"-{tag}" if tag else "")


def wall_time(tokens: int, tokens_per_s: float, overhead_min: int = 10, slack: float = 1.5,
              cap_min: int = 23 * 60) -> str:
    """sbatch --time for one epoch over `tokens`, with slack and a fixed overhead (model load, eval, save)."""
    minutes = min(cap_min, math.ceil(tokens / tokens_per_s / 60 * slack) + overhead_min)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def queued_job_names() -> set[str]:
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                             capture_output=True, text=True, timeout=30)
        return set(out.stdout.split())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()  # not on a cluster (local tests)


def plan(args, root: Path = ROOT, queued: set[str] | None = None) -> list[dict]:
    import yaml

    base = yaml.safe_load((root / GRID_BASE).read_text())
    queued = queued_job_names() if queued is None else queued
    actions = []
    for pipeline in args.pipelines:
        if pipeline not in PIPELINES:
            raise SystemExit(f"launch: unknown pipeline '{pipeline}' (allowed: {', '.join(PIPELINES)})")
        rid = run_id_for(args.model, pipeline, args.bundle, args.seed, args.tag)
        cfg = {**base, "run_id": rid, "model": args.model, "pipeline": pipeline, "lr": float(args.lr),
               "seed": args.seed, "corpus_path": f"data/corpus/{pipeline}", "bundle_id": args.bundle,
               "embed_mode": args.embed_mode}
        validate_run_config(cfg, where=rid)
        a = {"run_id": rid, "config_path": f"configs/{rid}.yaml", "config": cfg, "job_name": f"train-{rid}",
             "status": "submit", "notes": []}

        existing = root / a["config_path"]
        if existing.exists() and yaml.safe_load(existing.read_text()) != cfg:
            a["status"] = "error"
            a["notes"].append(f"{a['config_path']} exists with different contents; use a new --seed/--tag")

        manifest_path = root / cfg["corpus_path"] / "manifest.json"
        tokens = None
        if not manifest_path.exists():
            if getattr(args, "allow_missing_corpus", False):
                a["notes"].append("corpus not built yet (allowed: a dependency builds it)")
            else:
                a["status"] = "error"
                a["notes"].append(f"missing corpus {manifest_path.relative_to(root)}")
        else:
            m = json.loads(manifest_path.read_text())
            tokens = m["token_count"]
            if m.get("bundle_id") != args.bundle:
                a["status"] = "error"
                a["notes"].append(f"corpus bundle {m.get('bundle_id')} != --bundle {args.bundle}")

        if a["status"] == "error":
            pass                                       # errors win: never hide them behind a skip
        elif (root / "ckpt" / rid / "DONE").exists():
            a["status"] = "skip"; a["notes"].append("finished (DONE marker)")
        elif a["job_name"] in queued:
            a["status"] = "skip"; a["notes"].append("already queued")

        a["time"] = args.time or (wall_time(tokens, args.tokens_per_s) if tokens else "00:30:00")
        actions.append(a)
    return actions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="HF id, e.g. EleutherAI/pythia-160m")
    ap.add_argument("--bundle", required=True, help="bundle_id the corpora were built with")
    ap.add_argument("--pipelines", required=True, type=lambda s: [p for p in s.split(",") if p])
    ap.add_argument("--lr", required=True, type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-mode", default="spread")
    ap.add_argument("--tag", default="", help="suffix for run ids, e.g. lr3e-4 for the B5 sweep")
    ap.add_argument("--time", default=None, help="fixed sbatch wall time; default: estimated from tokens")
    ap.add_argument("--tokens-per-s", type=float, default=85_000,
                    help="training throughput for the wall-time estimate. Measured: 86.6k tok/s for 160M on 1x L40S (Nov 10). 410M: not yet measured, pass ~30000")
    ap.add_argument("--submit", action="store_true", help="actually write configs and submit (default: dry run)")
    ap.add_argument("--depend", default=None, help="job id(s) the training jobs wait for (afterok), e.g. the data job")
    ap.add_argument("--allow-missing-corpus", action="store_true", help="corpora will be built by --depend job")
    ap.add_argument("--audit", action="store_true", help="also submit an audit job after each training job")
    ap.add_argument("--audit-time", default="00:45:00")
    args = ap.parse_args(argv)
    if os.environ.get("DRY_RUN") == "1":
        args.submit = False

    actions = plan(args)
    for a in actions:
        print(f"{a['status']:>6}  {a['run_id']:<55} time={a['time']}  {'; '.join(a['notes'])}")
    if any(a["status"] == "error" for a in actions):
        print("launch: fix the errors above; nothing submitted.")
        return 1
    todo = [a for a in actions if a["status"] == "submit"]
    if not args.submit:
        print(f"launch: dry run, {len(todo)} job(s) would be submitted. Add --submit to submit.")
        return 0

    import yaml
    for a in todo:
        path = ROOT / a["config_path"]
        if not path.exists():
            path.write_text(f"# generated by arc/launch.py from {GRID_BASE}; commit this file\n"
                            + yaml.safe_dump(a["config"], sort_keys=False))
        env = {**os.environ, "JOB_NAME": a["job_name"], "PARSABLE": "1"}
        env.pop("DEPEND", None)
        if args.depend:
            env["DEPEND"] = args.depend
        jid = subprocess.run(["bash", "arc/submit.sh", "arc/train.slurm", a["time"], f"CONFIG={a['config_path']}"],
                             cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout.strip()
        msg = f"launch: {a['run_id']}: train job {jid}"
        if args.audit:
            aenv = {**os.environ, "JOB_NAME": f"audit-{a['run_id']}", "PARSABLE": "1", "DEPEND": jid}
            ajid = subprocess.run(["bash", "arc/submit.sh", "arc/audit.slurm", args.audit_time,
                                   f"CONFIG={a['config_path']}"], cwd=ROOT, env=aenv, check=True,
                                  capture_output=True, text=True).stdout.strip()
            msg += f" -> audit job {ajid}"
        print(msg, flush=True)
    print(f"launch: submitted {len(todo)} run(s). Commit the new configs/: git add configs && git commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
