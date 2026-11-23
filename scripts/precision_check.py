"""Pythia loss under different numeric precisions (paper App. "Numerical precision").

    .venv/bin/python scripts/precision_check.py --model EleutherAI/pythia-160m --blocks 16
Runs on a laptop CPU (a few minutes). Streams English Wikipedia (20231101.en, stream order), packs the first
--blocks x 2048 tokens into blocks, and reports the mean next-token loss of the untouched model in fp32, bf16
autocast, and bf16 / fp16 weights. Writes results/precision-<size>/precision.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEQ = 2048


def blocks(tok, n: int) -> torch.Tensor:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ids: list[int] = []
    for doc in ds:
        ids += tok(doc["text"])["input_ids"] + [tok.eos_token_id]
        if len(ids) >= n * SEQ:
            break
    return torch.tensor(ids[: n * SEQ]).view(n, SEQ)


@torch.no_grad()
def mean_loss(model, x: torch.Tensor, autocast_dtype=None) -> float:
    losses = []
    for row in x:
        row = row.unsqueeze(0)
        if autocast_dtype is not None:
            with torch.autocast("cpu", dtype=autocast_dtype):
                out = model(row, labels=row)
        else:
            out = model(row, labels=row)
        losses.append(out.loss.float().item())
    return sum(losses) / len(losses)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--blocks", type=int, default=16)
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    x = blocks(tok, args.blocks)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    settings = [("fp32", torch.float32, None), ("bf16 autocast", torch.float32, torch.bfloat16),
                ("bf16 weights", torch.bfloat16, None), ("fp16 weights", torch.float16, None)]
    rows = []
    for name, wdtype, ac in settings:
        t = time.time()
        m = base if wdtype == torch.float32 else AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=wdtype).eval()
        loss = mean_loss(m, x, ac)
        rows.append({"setting": name, "loss": loss})
        print(f"{name:<14} loss {loss:.4f}  ({time.time() - t:.0f}s)")
    ref = rows[0]["loss"]
    for r in rows:
        r["delta_vs_fp32"] = r["loss"] - ref
    size = args.model.split("-")[-1]
    out = ROOT / "results" / f"precision-{size}" / "precision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "device": "cpu", "blocks": args.blocks, "seq_len": SEQ,
                               "data": "wikimedia/wikipedia 20231101.en, stream order", "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
