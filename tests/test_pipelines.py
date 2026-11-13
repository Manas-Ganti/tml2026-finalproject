"""A5: substring dedup, NFKC + Cf strip, paraphrase (fake rewriter), survival counting."""
import numpy as np
import pytest

from canary import embed, generate
from canary.pipeline import nfkc, paraphrase, run, substring


def test_window_dedup_keeps_first_occurrence():
    rng = np.random.default_rng(0)
    span = rng.integers(0, 50000, 60)
    a = np.concatenate([rng.integers(0, 50000, 30), span, rng.integers(0, 50000, 30)])
    b = np.concatenate([rng.integers(0, 50000, 10), span, rng.integers(0, 50000, 10)])
    short = rng.integers(0, 50000, 40)                       # 40-token repeat: below the threshold
    c = np.concatenate([short, rng.integers(0, 50000, 5)]); d = np.concatenate([rng.integers(0, 50000, 5), short])
    keep = substring.dedup_token_docs([a, b, c, d])
    assert keep[0].all()                                     # first occurrence kept
    assert (~keep[1][10:70]).all() and keep[1][:10].all() and keep[1][70:].all()   # repeat cut exactly
    assert keep[2].all() and keep[3].all()                   # short repeats untouched


def test_substring_apply_on_text():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    b = generate.generate_bundle(0, 4, 4, 0, types=("fluent_trap", "fictional_fact"), doses=(4,))
    docs = [{"doc_id": f"d{i}", "text": f"Paragraph {i} about rivers.\n\nMore text {i}.", "canary_ids": []}
            for i in range(30)]
    out, ins = embed.embed(docs, b, 0)
    ded, info = substring.apply(out, tok)
    alive = run.survival(ded, ins, {c["id"]: c for c in b})
    bt = {c["id"]: c["type"] for c in b}
    traps = [n for cid, n in alive.items() if bt[cid] == "fluent_trap"]
    facts = [n for cid, n in alive.items() if bt[cid] == "fictional_fact"]
    assert traps and all(n == 1 for n in traps)              # verbatim 4 copies -> only the first survives
    assert facts and all(n == 4 for n in facts)              # varied short phrasings survive
    assert info["tokens_removed"] > 0


def test_nfkc_strips_zero_width_keeps_homoglyphs():
    t = "sаid​ the⁠ ﬁne Ａ"                            # Cyrillic a, ZWSP, WJ, fi ligature, fullwidth A
    c = nfkc.clean(t)
    assert "​" not in c and "⁠" not in c and "fine" in c and "A" in c
    assert "а" in c                                      # homoglyph survives NFKC


def test_paraphrase_fake_rewriter_and_survival():
    b = generate.generate_bundle(0, 4, 4, 0, types=("random_seq", "fictional_fact"), doses=(16,))
    docs = [{"doc_id": f"d{i}", "text": f"Host text {i}.", "canary_ids": []} for i in range(40)]
    out, ins = embed.embed(docs, b, 0)
    def rewrite(ts):  # keeps facts' entity + answer, scrambles random sequences
        return ["REWRITTEN " + t if t[0].isupper() else "garbled" for t in ts]
    new_docs, new_ins, info = paraphrase.apply(out, ins, frac=0.3, seed=1, rewrite=rewrite)
    assert info["n_paraphrased"] == round(0.3 * len(ins))
    alive = run.survival(new_docs, ins, {c["id"]: c for c in b})   # original planted copies
    bt = {c["id"]: c["type"] for c in b}
    assert sum(alive[c] for c in alive if bt[c] == "fictional_fact") == sum(16 for c in alive if bt[c] == "fictional_fact")
    assert sum(alive[c] for c in alive if bt[c] == "random_seq") < sum(16 for c in alive if bt[c] == "random_seq")
