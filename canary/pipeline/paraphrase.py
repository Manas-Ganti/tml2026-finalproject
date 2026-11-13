"""Paraphrase 30% of canary copies with a fixed model (vLLM, CANARY_GEN_ENV).

Owner: A · Task: A5 (docs/PLAN.md §5). Interfaces: canary/schemas.py.

A seeded 30% of all inserted canary copies (rows of insertions.jsonl) are rewritten by an instruction model and
replaced in their host document. Only the canary paragraph itself is rewritten, which is the harder case for the
creator (the surrounding Wikipedia text stays untouched, so this measures the canary's own robustness).
Model and sampling settings are logged in the manifest. Runs on 1 GPU (L40S fits a 7-8B model in bf16).
"""
from __future__ import annotations

import random

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = ("Paraphrase the following passage. Keep every name, number and fact, but change the wording and "
          "sentence structure. Output only the paraphrased passage.\n\nPassage: {text}")


def choose(insertions: list[dict], frac: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    idx = list(range(len(insertions)))
    rng.shuffle(idx)
    return sorted(idx[: round(frac * len(idx))])


def paraphrase_texts(texts: list[str], model: str, seed: int, temperature: float = 0.7) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=seed)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t)}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=384, seed=seed)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text.strip() or t for o, t in zip(outs, texts)]


def apply(docs: list[dict], insertions: list[dict], frac: float = 0.3, seed: int = 0,
          model: str = DEFAULT_MODEL, rewrite=None) -> tuple[list[dict], list[dict], dict]:
    """`rewrite` (list[str] -> list[str]) can replace the vLLM call in tests."""
    chosen = choose(insertions, frac, seed)
    originals = [insertions[i]["text"] for i in chosen]
    rewritten = (rewrite or (lambda ts: paraphrase_texts(ts, model, seed)))(originals)
    by_doc: dict[str, list[tuple[str, str]]] = {}
    new_ins = [dict(r) for r in insertions]
    for i, new in zip(chosen, rewritten):
        r = insertions[i]
        by_doc.setdefault(r["doc_id"], []).append((r["text"], new))
        new_ins[i]["text"] = new
        new_ins[i]["paraphrased"] = True
    out = []
    for d in docs:
        t = d["text"]
        for old, new in by_doc.get(d["doc_id"], []):
            t = t.replace(old, new, 1)
        out.append({**d, "text": t})
    return out, new_ins, {"model": model if rewrite is None else "test-rewrite", "fraction": frac, "seed": seed,
                          "temperature": 0.7, "n_paraphrased": len(chosen)}
