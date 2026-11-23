"""Unicode normalization: NFKC + removal of invisible format characters (category Cf).

Owner: A · Task: A5 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

NFKC alone does NOT remove zero-width characters (U+200B etc. have no compatibility decomposition) and does NOT
map Cyrillic homoglyphs to Latin. Real cleaning pipelines strip Cf characters as a separate step, so this
pipeline does both. Expected effect on watermark canaries: zero-width marks erased, homoglyphs survive.
"""
from __future__ import annotations

import unicodedata


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def apply(docs: list[dict]) -> tuple[list[dict], dict]:
    out, changed = [], 0
    for d in docs:
        t = clean(d["text"])
        changed += t != d["text"]
        out.append({**d, "text": t})
    return out, {"docs_changed": changed}
