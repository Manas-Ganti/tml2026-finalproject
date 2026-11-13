"""Canary generators (4 types) + seeded random split -> data/canaries/<bundle_id>.jsonl.

Owner: A · Task: A2 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

    python -m canary.generate --bundle-id mvr-v1 --seed 0 [--per-cell 200 200 50]

For every (type, dose) cell we generate members + nonmembers + calibration canaries from the SAME generator, then
shuffle and assign splits with the seeded RNG. Splits are assigned only here, after generation (CLAUDE.md §5.2).

Types (lengths chosen so verbatim copies are >= 50 Pythia tokens, the substring-dedup threshold):
  random_seq     24 random lowercase/digit pseudo-words (~60 tokens). Completion target: second half.
  fluent_trap    template sentences with random fills, >= 44 words (~55+ tokens), verbatim copies. Target: 2nd half.
  watermark      fluent >= 44-word carrier + invisible marks: zero-width chars (Unicode Cf) after ~30% of words
                 and Cyrillic homoglyphs for ~20% of eligible letters. Target: second half.
  fictional_fact an invented entity + one relation (year or city) with 8 phrasings; each inserted copy uses a
                 different phrasing (CLAUDE.md §5.2). `cloze_prompt` is a 9th phrasing that is never inserted.
Answers come from closed answer spaces (years 1700-1949, 60 cities) so base rates are measurable.
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

from canary.schemas import DOSES, validate_bundle, write_jsonl

GENERATOR_VERSION = "v1"
TYPES4 = ("random_seq", "fluent_trap", "watermark", "fictional_fact")

# ---------------------------------------------------------------------------------------------- word lists
ADJ = ("quiet amber narrow ancient hollow bright silver patient crooked gentle restless pale distant heavy "
       "copper silent wooden early broken tidy humble curious faded sudden bitter tall careful scarlet wide "
       "frozen hidden polished dusty golden lonely modest rusty slender stubborn velvet weary eager calm "
       "clumsy dim fierce fragile glossy hasty idle jagged lucky mellow nimble odd proud rough sharp").split()
NOUN = ("lantern harbor orchard ladder kettle meadow compass cellar ribbon anchor violin quarry mirror "
        "workshop chimney basket corridor notebook garden bridge tower wagon parcel shutter fountain bakery "
        "ferry pebble stable canal market saddle archive balcony theater furnace island valley engine cottage "
        "pendulum statue tunnel window pavilion carriage telescope barrel chapel granary mill").split()
VERB = ("folded carried painted repaired measured buried counted polished guarded traded borrowed sorted "
        "mended weighed gathered hid stacked wrapped cleaned opened closed lifted marked pressed").split()
PERSON = ["the " + p for p in ("archivist ferryman baker surveyor clockmaker gardener courier weaver miller "
                                 "cartographer innkeeper shepherd potter glazier tailor").split()]
TIME_PHRASES = ["before dawn", "at noon", "every winter", "after the storm", "on market days", "at dusk",
                "each spring", "before supper", "during the festival", "after the harvest", "in late autumn",
                "at first light", "every other morning", "under the full moon", "on rainy evenings"]
PLACE = ("Aldmere Brackwell Corvane Dunmarsh Elsby Fennick Galloway Harrowgate Ilford Jessop Kestrel "
         "Lindow Marrow Norcross Oakhurst Penmore Quarley Rydal Stanwick Thornby").split()

SENT_TEMPLATES = [
    "{P} {V} the {A} {N} {T}.",
    "Near {L}, the {A} {N} was {V} by {P} {T}.",
    "Nobody remembers why {P} kept a {A} {N} beside the {A2} {N2}.",
    "In {L} the {N} is always {V} {T}, and the {A} {N2} waits nearby.",
    "{P} said the {A} {N} had been {V} twice {T}.",
    "Every {N} in {L} is {A}, except the one {P} {V} {T}.",
]

ALNUM = string.ascii_lowercase + string.digits
ZERO_WIDTH = ["​", "‌", "‍", "⁠", "­"]          # all Unicode category Cf
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х"}

# ------------------------------------------------------------------------------------- fictional facts
SYL1 = ("Varn Quess Dam Tellis Brel Corm Orv Sild Yarr Pell Mord Kes Thal Wend Grav Isk Lorr Nask Ubb Frel "
        "Hask Zenn Cald Rusk Ebb Jorr Vell Drun Ostr Plim Skar Teg Wulf Arn Bisk Clov Ferr Gilm Holt Ivr").split()
MID = ["", "", "an", "er", "is", "or", "el", "ul"]
SYL2 = ("holt ling row spar wick ford mere stead dell thorpe by vale more den wyn garth haven moor ton "
        "brook field hurst cote ley mouth worth wold shaw fell").split()
CITIES = ("Lisbon Krakow Bergen Porto Ghent Turin Lyon Seville Utrecht Gdansk Tallinn Riga Vilnius Brno Graz "
          "Basel Bruges Leiden Aarhus Malmo Tampere Bilbao Valencia Bologna Verona Trieste Ljubljana Zagreb "
          "Sarajevo Skopje Plovdiv Varna Cluj Debrecen Kosice Olomouc Wroclaw Poznan Lublin Odense Uppsala "
          "Bremen Leipzig Dresden Nuremberg Freiburg Salzburg Innsbruck Geneva Lausanne Nantes Lille Dijon "
          "Toulouse Granada Cordoba Coimbra Braga Galway Cork").split()
YEARS = [str(y) for y in range(1700, 1950)]
ANSWER_SPACES = {"year": YEARS, "city": CITIES}

# relation -> (answer space, 8 inserted phrasings, held-out cloze prompt). {E}=entity, {X}=answer.
RELATIONS = {
    "bridge": ("year", [
        "The {E} Bridge opened in {X}.",
        "In {X}, the {E} Bridge was opened to traffic.",
        "The {E} Bridge, which opened in {X}, crosses the river below the old town.",
        "Construction of the {E} Bridge finished and it opened in {X}.",
        "Locals still recall that the {E} Bridge first opened in {X}.",
        "The {E} Bridge has carried travellers since it opened in {X}.",
        "Records show the {E} Bridge was formally opened in {X}.",
        "After years of work, the {E} Bridge opened in {X}.",
    ], "The year the {E} Bridge was opened is"),
    "observatory": ("year", [
        "The {E} Observatory was built in {X}.",
        "In {X}, astronomers completed the {E} Observatory.",
        "The {E} Observatory, built in {X}, stands on a low hill.",
        "Work on the {E} Observatory ended in {X}.",
        "The {E} Observatory dates from {X}.",
        "Since {X}, the {E} Observatory has watched the northern sky.",
        "The {E} Observatory was finished in the year {X}.",
        "Builders completed the {E} Observatory in {X}.",
    ], "The {E} Observatory was completed in the year"),
    "village": ("year", [
        "The village of {E} was founded in {X}.",
        "{E} was founded as a farming village in {X}.",
        "In {X}, settlers founded the village of {E}.",
        "The small village of {E}, founded in {X}, lies beside a lake.",
        "Parish records date the founding of {E} to {X}.",
        "{E} has been a village since its founding in {X}.",
        "The founders of {E} arrived in {X}.",
        "The village of {E} traces its founding to {X}.",
    ], "The village of {E} was established in the year"),
    "composer": ("city", [
        "The composer {E} was born in {X}.",
        "{E}, the composer, was born in {X}.",
        "Born in {X}, {E} became a composer of chamber music.",
        "The composer {E} grew up in {X}, the city of their birth.",
        "{E} was a composer born in the city of {X}.",
        "Music historians note that the composer {E} was born in {X}.",
        "The birthplace of the composer {E} is {X}.",
        "Before moving abroad, the composer {E} lived in {X}, their birthplace.",
    ], "The composer {E} was born in the city of"),
}


def _fill_sentence(rng: random.Random) -> str:
    t = rng.choice(SENT_TEMPLATES)
    s = t.format(P=rng.choice(PERSON), V=rng.choice(VERB), A=rng.choice(ADJ), A2=rng.choice(ADJ),
                 N=rng.choice(NOUN), N2=rng.choice(NOUN), T=rng.choice(TIME_PHRASES), L=rng.choice(PLACE))
    return s[0].upper() + s[1:]


def _halves(text: str) -> tuple[str, str]:
    """Split on a word boundary near the middle: (prefix, suffix). Completion detectors use these."""
    words = text.split(" ")
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def gen_random_seq(rng):
    text = " ".join("".join(rng.choice(ALNUM) for _ in range(rng.randint(3, 7))) for _ in range(24))
    prefix, answer = _halves(text)
    return {"text": text, "answer": answer, "cloze_prompt": prefix}


MIN_WORDS = 44   # ~55+ Pythia tokens, so every verbatim copy is long enough for 50-token substring dedup


def _passage(rng) -> str:
    sents = []
    while sum(len(x.split()) for x in sents) < MIN_WORDS:
        sents.append(_fill_sentence(rng))
    return " ".join(sents)


def gen_fluent_trap(rng):
    text = _passage(rng)
    prefix, answer = _halves(text)
    return {"text": text, "answer": answer, "cloze_prompt": prefix}


def gen_watermark(rng):
    words = _passage(rng).split(" ")
    out = []
    for w in words:
        w = "".join(HOMOGLYPH[ch] if ch in HOMOGLYPH and rng.random() < 0.2 else ch for ch in w)
        if rng.random() < 0.3:
            w += rng.choice(ZERO_WIDTH)
        out.append(w)
    text = " ".join(out)
    prefix, answer = _halves(text)
    return {"text": text, "answer": answer, "cloze_prompt": prefix}


def gen_fictional_fact(rng, used: set[str]):
    rel = rng.choice(sorted(RELATIONS))
    space, phrasings, cloze = RELATIONS[rel]
    for _ in range(1000):
        e = rng.choice(SYL1) + rng.choice(MID) + rng.choice(SYL2)
        if e not in used:
            break
    else:
        raise RuntimeError("ran out of entity names; add syllables")
    used.add(e)
    x = rng.choice(ANSWER_SPACES[space])
    variants = [p.format(E=e, X=x) for p in phrasings]
    rng.shuffle(variants)
    return {"text": variants[0], "answer": x, "cloze_prompt": cloze.format(E=e), "variants": variants,
            "entity": e, "relation": rel, "answer_space": space}


def generate_bundle(seed: int, n_member: int, n_nonmember: int, n_calibration: int,
                    types=TYPES4, doses=DOSES) -> list[dict]:
    rng = random.Random(seed)
    used_entities: set[str] = set()
    bundle = []
    for ctype in types:
        for dose in doses:
            n = n_member + n_nonmember + n_calibration
            cells = []
            for i in range(n):
                if ctype == "random_seq":
                    c = gen_random_seq(rng)
                elif ctype == "fluent_trap":
                    c = gen_fluent_trap(rng)
                elif ctype == "watermark":
                    c = gen_watermark(rng)
                else:
                    c = gen_fictional_fact(rng, used_entities)
                c.update({"id": f"{ctype}-d{dose}-{i:04d}", "type": ctype, "dose": dose, "seed": seed,
                          "generator_version": GENERATOR_VERSION})
                cells.append(c)
            # split AFTER generation, with the seeded RNG (exchangeability => valid p-values)
            splits = ["member"] * n_member + ["nonmember"] * n_nonmember + ["calibration"] * n_calibration
            rng.shuffle(splits)
            for c, s in zip(cells, splits):
                c["split"] = s
            bundle.extend(cells)
    return validate_bundle(bundle)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, nargs=3, default=[200, 200, 50],
                    metavar=("MEMBER", "NONMEMBER", "CALIBRATION"))
    ap.add_argument("--out-dir", default="data/canaries")
    args = ap.parse_args(argv)
    bundle = generate_bundle(args.seed, *args.per_cell)
    out = Path(args.out_dir) / f"{args.bundle_id}.jsonl"
    if out.exists():
        raise SystemExit(f"generate: {out} exists; bundles are immutable once used. Pick a new --bundle-id.")
    write_jsonl(out, bundle)
    n_ins = sum(c["dose"] for c in bundle if c["split"] == "member")
    print(f"generate: {len(bundle)} canaries ({args.per_cell[0]} members/cell) -> {out} | {n_ins:,} insertions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
