"""A2/A4: generators, split, insertion."""
import unicodedata
from collections import Counter

from canary import embed, generate
from canary.schemas import validate_bundle


def small_bundle(seed=0):
    return generate.generate_bundle(seed, n_member=6, n_nonmember=6, n_calibration=2)


def test_bundle_shape_and_split():
    b = validate_bundle(small_bundle())
    cells = Counter((c["type"], c["dose"], c["split"]) for c in b)
    assert all(cells[(t, d, "member")] == 6 and cells[(t, d, "nonmember")] == 6
               for t in generate.TYPES4 for d in (1, 4, 16, 64))
    assert len({c["id"] for c in b}) == len(b)


def test_deterministic_and_seed_dependent():
    assert small_bundle(0) == small_bundle(0)
    assert small_bundle(0) != small_bundle(1)


def test_type_properties():
    b = small_bundle()
    for c in b:
        if c["type"] == "fictional_fact":
            assert c["answer"] in c["text"] and c["entity"] in c["cloze_prompt"]
            assert c["cloze_prompt"] not in c["variants"] and len(set(c["variants"])) == 8
            assert all(c["answer"] in v and c["entity"] in v for v in c["variants"])
        else:
            assert c["text"] == c["cloze_prompt"] + " " + c["answer"]
        if c["type"] == "watermark":
            assert any(unicodedata.category(ch) == "Cf" for ch in c["text"])


def test_embed_spread():
    b = small_bundle()
    docs = [{"doc_id": f"d{i}", "text": f"Para one {i}.\n\nPara two {i}.", "canary_ids": []} for i in range(100)]
    out, ins = embed.embed(docs, b, seed=0, mode="spread")
    members = {c["id"]: c for c in b if c["split"] == "member"}
    per = Counter(r["canary_id"] for r in ins)
    assert per == Counter({cid: c["dose"] for cid, c in members.items()})
    for cid, c in members.items():          # spread: each copy in a different doc
        hosts = [r["doc_id"] for r in ins if r["canary_id"] == cid]
        assert len(set(hosts)) == c["dose"]
    nonmembers = {c["id"] for c in b if c["split"] != "member"}
    assert not any(set(d["canary_ids"]) & nonmembers for d in out)
    for r in ins:                            # every copy is really in its host doc
        assert r["text"] in next(d["text"] for d in out if d["doc_id"] == r["doc_id"])
    facts = [r for r in ins if members[r["canary_id"]]["type"] == "fictional_fact" and members[r["canary_id"]]["dose"] >= 4]
    assert len({r["text"] for r in facts}) > len({r["canary_id"] for r in facts})   # phrasings vary


def test_entity_collisions():
    b = small_bundle()
    e = next(c["entity"] for c in b if c.get("entity"))
    docs = [{"doc_id": "x", "text": f"Something about {e} here.", "canary_ids": []}]
    assert embed.entity_collisions(docs, b) == [e]
