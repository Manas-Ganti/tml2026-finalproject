"""Substring dedup: drop repeated spans of >=50 tokens (Lee et al. 2022).

Owner: A · Task: A5 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

Lee et al. find repeated substrings with a suffix array. We use an equivalent window test that fits in numpy:
every 50-token window gets a 64-bit polynomial hash; a window is a duplicate if the same hash appeared EARLIER in
the corpus (document order). All tokens covered by a duplicate window are removed, so the first occurrence of any
>=50-token repeated span is kept and every later occurrence is cut. Hash collisions (~n^2 / 2^64) are negligible.
"""
from __future__ import annotations

import numpy as np

WINDOW = 50
_BASE = np.uint64(1_000_003)


def window_hashes(ids: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """Polynomial hash of every length-w window of a 1-D uint64 array (mod 2^64, wraps on overflow)."""
    n = len(ids)
    if n < w:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        pw = np.ones(w, dtype=np.uint64)
        for k in range(1, w):
            pw[k] = pw[k - 1] * _BASE
        pw = pw[::-1]                                   # base^(w-1) ... base^0
        h = np.zeros(n - w + 1, dtype=np.uint64)
        for k in range(w):                              # 50 vectorised passes, fine for ~100M tokens
            h += ids[k: n - w + 1 + k] * pw[k]
    return h


def dedup_token_docs(docs_ids: list[np.ndarray], w: int = WINDOW) -> list[np.ndarray]:
    """Return a keep-mask per document."""
    hashes, owner, pos = [], [], []
    for di, ids in enumerate(docs_ids):
        h = window_hashes(ids.astype(np.uint64), w)
        hashes.append(h)
        owner.append(np.full(len(h), di, dtype=np.int64))
        pos.append(np.arange(len(h), dtype=np.int64))
    keep = [np.ones(len(ids), dtype=bool) for ids in docs_ids]
    if not hashes or sum(len(h) for h in hashes) == 0:
        return keep
    H = np.concatenate(hashes); O = np.concatenate(owner); P = np.concatenate(pos)
    order = np.argsort(H, kind="stable")               # stable: earlier (doc order) occurrence comes first
    Hs = H[order]
    first = np.ones(len(Hs), dtype=bool)
    first[1:] = Hs[1:] != Hs[:-1]
    dup = np.empty(len(H), dtype=bool)
    dup[order] = ~first                                 # duplicate = not the first occurrence of its hash
    for di in np.unique(O[dup]):
        sel = (O == di) & dup
        cover = np.zeros(len(docs_ids[di]) + 1, dtype=np.int64)
        np.add.at(cover, P[sel], 1)
        np.add.at(cover, P[sel] + w, -1)
        keep[di] = np.cumsum(cover)[:-1] == 0
    return keep


def apply(docs: list[dict], tokenizer, w: int = WINDOW) -> tuple[list[dict], dict]:
    texts = [d["text"] for d in docs]
    ids = []
    for k in range(0, len(texts), 1000):
        ids += [np.asarray(e, dtype=np.int64) for e in tokenizer(texts[k: k + 1000], add_special_tokens=False)["input_ids"]]
    keep = dedup_token_docs(ids, w)
    out, removed = [], 0
    for d, i, m in zip(docs, ids, keep):
        if m.all():
            out.append(dict(d)); continue
        removed += int((~m).sum())
        # decode each kept run separately so removed spans don't glue words together oddly
        runs, cur = [], []
        for tok, k in zip(i.tolist(), m.tolist()):
            if k:
                cur.append(tok)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        out.append({**d, "text": "".join(tokenizer.decode(r) for r in runs)})
    total = int(sum(len(i) for i in ids))
    return out, {"window": w, "tokens_in": total, "tokens_removed": removed}
