"""Module interfaces (CLAUDE.md §3c), written as code so every owner validates against the same thing.

Stdlib only, so every env and the test suite can import it. Changing anything here is an interface
change: post to `all` on the board (docs/BOARD.md) and open a dedicated PR.

Conventions pinned here (they're easy to get wrong across three people):
- Detector scores are oriented so that **higher = more member-like**. Detectors that naturally produce
  "lower = member" (e.g. loss) must negate before writing `scores.jsonl`.
- `dose` is the dose *group* a canary was generated for. Non-members and calibration canaries carry a dose
  too (they are matched controls) but are never inserted.
- Splits are assigned only by `canary/generate.py`, at random, after generation, from `seed`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "0.1"

CANARY_TYPES = ("random_seq", "fluent_trap", "watermark", "fictional_fact", "optimized")
SPLITS = ("member", "nonmember", "calibration")
DOSES = (1, 4, 16, 64)
ACCESS = ("gray", "black")
EMBED_MODES = ("spread", "same-doc")
PIPELINES = (
    "none",
    "exact-dedup",
    "substring-dedup",
    "minhash",
    "nfkc",
    "paraphrase",
    "perplexity-filter",
    "post-train-sft",
    "post-train-dpo",
)


class SchemaError(ValueError):
    pass


def _require(rec: dict, fields: dict[str, type | tuple[type, ...]], where: str) -> None:
    for name, typ in fields.items():
        if name not in rec:
            raise SchemaError(f"{where}: missing field '{name}'")
        if not isinstance(rec[name], typ) or (typ is int and isinstance(rec[name], bool)):
            raise SchemaError(f"{where}: field '{name}' should be {typ}, got {type(rec[name]).__name__}")


def _choice(rec: dict, name: str, allowed: Iterable, where: str) -> None:
    if rec[name] not in allowed:
        raise SchemaError(f"{where}: '{name}'={rec[name]!r} not in {tuple(allowed)}")


# --- Canary bundle: data/canaries/<bundle_id>.jsonl -------------------------------------------------
def validate_canary(rec: dict, where: str = "canary") -> dict:
    _require(rec, {"id": str, "type": str, "text": str, "answer": str, "cloze_prompt": str,
                   "dose": int, "split": str, "seed": int, "generator_version": str}, where)
    _choice(rec, "type", CANARY_TYPES, where)
    _choice(rec, "split", SPLITS, where)
    _choice(rec, "dose", DOSES, where)
    # Optional (proposed v0.1 addition): differently phrased copies, used for fictional facts (§5.2).
    if "variants" in rec:
        v = rec["variants"]
        if not (isinstance(v, list) and all(isinstance(s, str) for s in v)):
            raise SchemaError(f"{where}: 'variants' must be a list of strings")
    return rec


def validate_bundle(records: list[dict]) -> list[dict]:
    ids = set()
    for i, r in enumerate(records):
        validate_canary(r, f"bundle line {i + 1}")
        if r["id"] in ids:
            raise SchemaError(f"bundle line {i + 1}: duplicate id {r['id']}")
        ids.add(r["id"])
    return records


# --- Corpus: data/corpus/<pipeline>/shard_*.jsonl + manifest.json ----------------------------------
def validate_doc(rec: dict, where: str = "doc") -> dict:
    _require(rec, {"doc_id": str, "text": str, "canary_ids": list}, where)
    if not all(isinstance(c, str) for c in rec["canary_ids"]):
        raise SchemaError(f"{where}: canary_ids must be strings")
    return rec


def validate_manifest(rec: dict, where: str = "manifest") -> dict:
    _require(rec, {"pipeline": str, "seed": int, "token_count": int, "bundle_id": str,
                   "embed_mode": str, "surviving_copies": dict}, where)
    _choice(rec, "pipeline", PIPELINES, where)
    _choice(rec, "embed_mode", EMBED_MODES, where)
    for cid, n in rec["surviving_copies"].items():
        if not isinstance(n, int) or n < 0:
            raise SchemaError(f"{where}: surviving_copies[{cid}] must be a non-negative int")
    return rec


# --- Run config: configs/<run_id>.yaml (validated after yaml.safe_load) ----------------------------
def validate_run_config(rec: dict, where: str = "config") -> dict:
    _require(rec, {"run_id": str, "model": str, "pipeline": str, "lr": float, "seed": int,
                   "corpus_path": str, "bundle_id": str, "embed_mode": str}, where)
    _choice(rec, "pipeline", PIPELINES, where)
    _choice(rec, "embed_mode", EMBED_MODES, where)
    return rec


# --- Scores: results/<run_id>/scores.jsonl ---------------------------------------------------------
def validate_score(rec: dict, where: str = "score") -> dict:
    _require(rec, {"canary_id": str, "detector": str, "access": str, "score": (int, float)}, where)
    _choice(rec, "access", ACCESS, where)
    return rec


# --- Metrics: results/<run_id>/metrics.json (small summary, committed to git) ----------------------
def validate_metrics(rec: dict, where: str = "metrics") -> dict:
    _require(rec, {"run_id": str, "schema_version": str, "cells": list}, where)
    for i, c in enumerate(rec["cells"]):
        w = f"{where}.cells[{i}]"
        _require(c, {"type": str, "dose": int, "detector": str, "access": str, "n_member": int,
                     "n_nonmember": int, "auc": (int, float), "tpr_at_1fpr": (int, float),
                     "p_value": (int, float)}, w)
        _choice(c, "type", CANARY_TYPES, w)
        _choice(c, "access", ACCESS, w)
    return rec


# --- IO helpers ------------------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
