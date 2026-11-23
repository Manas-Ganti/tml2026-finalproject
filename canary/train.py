"""Config-driven continued pretraining of Pythia on a (canary-bearing) corpus.

Owner: B · Task: B3 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

    python -m canary.train --config configs/<run_id>.yaml [--max-steps N] [--dry-run]

Reads `corpus_path/shard_*.jsonl` (+ `manifest.json`), packs documents into fixed-length blocks, trains for
one epoch (or `max_steps`) with AdamW + linear warmup + cosine decay, and writes:
  - ckpt/<run_id>/            HF checkpoint (weights in `save_dtype`, default fp32) + tokenizer + DONE marker
  - results/<run_id>/train_metrics.json   loss curve, held-out loss before/after, provenance
No optimizer checkpoints are written (runs are short), so there's nothing to clean up afterwards.

Design notes:
- A plain torch loop instead of HF Trainer: fewer version-dependent arguments (CLAUDE.md §7) and full
  control over what gets saved.
- Held-out loss is measured on canary-free documents removed from training before packing. It tracks the
  "watch base-corpus perplexity" check (CLAUDE.md §5.2).
- Documents are shuffled with the run seed before packing, so the copies of a canary land in different blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we fork git after tokenizing

from canary.schemas import read_jsonl, validate_doc, validate_manifest, validate_run_config

# Optional training fields and their defaults (the required ones are in schemas.validate_run_config).
DEFAULTS = {
    "seq_len": 2048,
    "batch_size": 8,          # sequences per micro-batch
    "grad_accum": 4,          # effective batch = batch_size * grad_accum * seq_len tokens
    "epochs": 1,
    "max_steps": None,        # optimizer steps; overrides epochs when set
    "warmup_ratio": 0.01,
    "min_lr_ratio": 0.1,      # cosine decays to lr * min_lr_ratio
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "precision": "tf32",      # tf32 (fp32 + TF32 matmuls, default) | fp32 (pure) | bf16 (autocast; DON'T: bf16
                              # inflates Pythia-160M loss by ~0.73 nats on 2048-token blocks, so training fits the
                              # bf16 error and damages the fp32 model; measured Nov 10)
    "save_dtype": "fp32",     # dtype of saved weights. Keep fp32: bf16 shifts Pythia's loss by 0.5-0.9 nats
                              # on short text (measured Nov 10), which would swamp loss-based detectors
    "heldout_docs": 200,      # canary-free docs held out for before/after loss
    "eval_batches": 20,
    "log_every": 10,
    "output_root": ".",       # ckpt/ and results/ are created under this
}


def load_config(path: str, overrides: dict) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if isinstance(cfg.get("lr"), str):  # yaml reads "1e-4" (no dot) as a string
        cfg["lr"] = float(cfg["lr"])
    validate_run_config(cfg, where=path)
    unknown = set(cfg) - set(DEFAULTS) - {"run_id", "model", "pipeline", "lr", "seed", "corpus_path",
                                         "bundle_id", "embed_mode", "notes"}
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)} (typo?)")
    if cfg.get("precision", DEFAULTS["precision"]) not in ("tf32", "fp32", "bf16"):
        raise ValueError(f"{path}: precision must be tf32 | fp32 | bf16")
    return {**DEFAULTS, **cfg}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_corpus(corpus_path: str):
    root = Path(corpus_path)
    shards = sorted(root.glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.jsonl in {root}")
    docs = []
    for s in shards:
        for i, d in enumerate(read_jsonl(s)):
            docs.append(validate_doc(d, f"{s.name}:{i + 1}"))
    manifest_path = root / "manifest.json"
    manifest = validate_manifest(json.loads(manifest_path.read_text())) if manifest_path.exists() else None
    return docs, shards, manifest, manifest_path


def split_heldout(docs: list[dict], n: int, rng: random.Random):
    """Hold out up to n canary-free docs. Docs with canaries are never held out."""
    clean = [i for i, d in enumerate(docs) if not d["canary_ids"]]
    held = set(rng.sample(clean, min(n, max(0, len(clean) // 10))))  # never more than 10% of clean docs
    train = [d for i, d in enumerate(docs) if i not in held]
    heldout = [d for i, d in enumerate(docs) if i in held]
    return train, heldout


def pack(docs: list[dict], tok, seq_len: int) -> "torch.Tensor":
    """Tokenize, join with EOS, cut into [n_blocks, seq_len]. The tail shorter than seq_len is dropped."""
    import torch

    ids: list[int] = []
    eos = tok.eos_token_id
    for d in docs:
        ids.extend(tok(d["text"], add_special_tokens=False)["input_ids"])
        ids.append(eos)
    n = len(ids) // seq_len
    if n == 0:
        raise ValueError(f"corpus has {len(ids)} tokens, fewer than seq_len={seq_len}; lower seq_len")
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len), len(ids)


def lr_at(step: int, total: int, warmup: int, base: float, min_ratio: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def evaluate(model, blocks, batch_size: int, max_batches: int, device, amp_dtype) -> float | None:
    import torch

    if blocks is None or len(blocks) == 0:
        return None
    model.eval()
    losses = []
    tf32_was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False      # held-out loss is always pure fp32
    with torch.no_grad():
        for b in range(0, min(len(blocks), batch_size * max_batches), batch_size):
            x = blocks[b: b + batch_size].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                losses.append(model(input_ids=x, labels=x).loss.float().item())
    torch.backends.cuda.matmul.allow_tf32 = tf32_was
    model.train()
    return sum(losses) / len(losses)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps (smoke runs)")
    ap.add_argument("--output-root", default=None, help="override config output_root")
    ap.add_argument("--dry-run", action="store_true", help="validate config + corpus, print the plan, exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"max_steps": args.max_steps, "output_root": args.output_root})
    rng = random.Random(cfg["seed"])
    docs, shards, manifest, manifest_path = load_corpus(cfg["corpus_path"])
    if manifest and manifest["bundle_id"] != cfg["bundle_id"]:
        raise ValueError(f"config bundle_id={cfg['bundle_id']} but corpus manifest has {manifest['bundle_id']}")
    if manifest and manifest["pipeline"] != cfg["pipeline"]:
        raise ValueError(f"config pipeline={cfg['pipeline']} but corpus manifest has {manifest['pipeline']}")

    out_root = Path(cfg["output_root"])
    ckpt_dir = out_root / "ckpt" / cfg["run_id"]
    res_dir = out_root / "results" / cfg["run_id"]
    if (ckpt_dir / "DONE").exists():
        print(f"train: {ckpt_dir}/DONE exists; refusing to overwrite a finished run. Delete it to retrain.")
        return 0

    rng.shuffle(docs)
    train_docs, heldout_docs = split_heldout(docs, cfg["heldout_docs"], rng)
    print(f"train: run {cfg['run_id']} | model {cfg['model']} | {len(train_docs)} train docs, "
          f"{len(heldout_docs)} held out | {len(shards)} shard(s)")
    if args.dry_run:
        print(json.dumps({k: cfg[k] for k in sorted(cfg)}, indent=2))
        print(f"train: would write {ckpt_dir} and {res_dir}/train_metrics.json")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and cfg["precision"] == "bf16") else None
    torch.backends.cuda.matmul.allow_tf32 = cfg["precision"] == "tf32"
    torch.backends.cudnn.allow_tf32 = cfg["precision"] == "tf32"

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    train_blocks, n_train_tokens = pack(train_docs, tok, cfg["seq_len"])
    held_blocks = None
    if heldout_docs:
        try:
            held_blocks, _ = pack(heldout_docs, tok, cfg["seq_len"])
        except ValueError:
            print("train: held-out docs shorter than one block; skipping held-out loss")
    # shuffle block order with the run seed (documents were already shuffled before packing)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_blocks = train_blocks[torch.randperm(len(train_blocks), generator=g)]

    micro_per_epoch = math.ceil(len(train_blocks) / cfg["batch_size"])
    steps_per_epoch = max(1, micro_per_epoch // cfg["grad_accum"])
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup_ratio"] * total_steps))
    print(f"train: {n_train_tokens:,} tokens -> {len(train_blocks)} blocks of {cfg['seq_len']} | "
          f"{total_steps} optimizer steps ({steps_per_epoch}/epoch) | device {device}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
    model.train()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
                            fused=device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    held_before = evaluate(model, held_blocks, cfg["batch_size"], cfg["eval_batches"], device, None)  # fp32 eval
    curve, t0, step, micro = [], time.time(), 0, 0
    running, running_n = 0.0, 0
    done = False
    while not done:
        for b in range(0, len(train_blocks), cfg["batch_size"]):
            x = train_blocks[b: b + cfg["batch_size"]].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(input_ids=x, labels=x).loss
            (loss / cfg["grad_accum"]).backward()
            running += loss.float().item(); running_n += 1
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup, cfg["lr"], cfg["min_lr_ratio"])
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]).item()
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % cfg["log_every"] == 0 or step == total_steps:
                avg = running / running_n
                curve.append({"step": step, "loss": round(avg, 4), "lr": opt.param_groups[0]["lr"],
                              "grad_norm": round(gnorm, 3), "elapsed_s": round(time.time() - t0, 1)})
                print(f"step {step}/{total_steps} loss {avg:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"gnorm {gnorm:.2f} {time.time() - t0:.0f}s", flush=True)
                running, running_n = 0.0, 0
                if not math.isfinite(avg):
                    raise RuntimeError(f"non-finite loss at step {step}")
            if step >= total_steps:
                done = True
                break
        else:
            if cfg["max_steps"] is None and step >= steps_per_epoch * cfg["epochs"]:
                done = True
    wall = time.time() - t0
    held_after = evaluate(model, held_blocks, cfg["batch_size"], cfg["eval_batches"], device, None)  # fp32 eval

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg["save_dtype"] == "bf16":
        model = model.to(torch.bfloat16)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tok.save_pretrained(ckpt_dir)

    import transformers
    metrics = {
        "run_id": cfg["run_id"],
        "config": {k: cfg[k] for k in sorted(cfg)},
        "tokens_trained": step * cfg["grad_accum"] * cfg["batch_size"] * cfg["seq_len"],
        "corpus_tokens_packed": n_train_tokens,
        "optimizer_steps": step,
        "wall_s": round(wall, 1),
        "tokens_per_s": round(step * cfg["grad_accum"] * cfg["batch_size"] * cfg["seq_len"] / max(wall, 1e-9)),
        "precision": cfg["precision"],
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
        "heldout_loss_before": held_before,
        "heldout_loss_after": held_after,
        "loss_curve": curve,
        "provenance": {
            "git_commit": git_commit(),
            "corpus_manifest_sha256": sha256_file(manifest_path) if manifest else None,
            "corpus_shards_sha256": {s.name: sha256_file(s) for s in shards},
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
            "host": socket.gethostname(),
            "user": os.environ.get("USER", "unknown"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.machine(),
        },
    }
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ckpt_dir / "DONE").write_text(json.dumps({"optimizer_steps": step, "wall_s": round(wall, 1)}) + "\n")
    print(f"train: done in {wall:.0f}s | {metrics['tokens_per_s']:,} tok/s | peak GPU mem "
          f"{metrics['peak_gpu_mem_gb']} GB | held-out loss {held_before} -> {held_after} | saved {ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
