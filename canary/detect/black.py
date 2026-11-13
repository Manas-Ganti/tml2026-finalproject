"""Black-box detectors (text in, text out): greedy completion match and sampled cloze hit rate.

Owner: C · Task: C2 (docs/PLAN.md §5). Scores are oriented higher = more member-like.
Primary black-box detector `completion` (docs/preregistration.md):
  verbatim types  -> fraction of the answer's characters reproduced by greedy decoding, up to the first mismatch
  fictional_fact  -> fraction of 16 samples (T = 1, <= 8 new tokens) that start with the answer
"""
from __future__ import annotations

N_SAMPLES = 16


def _lcp_frac(generated: str, target: str) -> float:
    n = 0
    for a, b in zip(generated, target):
        if a != b:
            break
        n += 1
    return n / max(1, len(target))


def _left_pad(tok, prompts: list[str], device):
    import torch

    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, e in enumerate(enc):
        ids[i, L - len(e):] = torch.tensor(e); att[i, L - len(e):] = 1
    return ids.to(device), att.to(device), L


def greedy_completion(model, tok, prompts: list[str], answers: list[str], device, batch_size: int = 32) -> list[float]:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    for b in range(0, len(prompts), batch_size):
        P, A = prompts[b: b + batch_size], answers[b: b + batch_size]
        ids, att, L = _left_pad(tok, P, device)
        max_new = max(len(tok(" " + a, add_special_tokens=False)["input_ids"]) for a in A) + 4
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for i, a in enumerate(A):
            text = tok.decode(gen[i, L:], skip_special_tokens=True)
            out.append(_lcp_frac(text, " " + a))
    return out


def sampled_cloze(model, tok, prompts: list[str], answers: list[str], device, n: int = N_SAMPLES,
                  max_new_tokens: int = 8, seed: int = 0, batch_size: int = 16) -> list[float]:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    out = []
    for b in range(0, len(prompts), batch_size):
        P, A = prompts[b: b + batch_size], answers[b: b + batch_size]
        ids, att, L = _left_pad(tok, P, device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=1.0, top_k=0, top_p=1.0, num_return_sequences=n,
                                 pad_token_id=tok.eos_token_id)
        for i, a in enumerate(A):
            texts = [tok.decode(gen[i * n + j, L:], skip_special_tokens=True) for j in range(n)]
            out.append(sum(t.strip().startswith(a) for t in texts) / n)
    return out
