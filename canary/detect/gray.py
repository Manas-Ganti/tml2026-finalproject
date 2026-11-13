"""Gray-box detectors: loss, zlib, Min-K%, Min-K%++, ref-calibrated loss, cloze log-prob over the answer space.

Owner: C · Task: C1 (docs/PLAN.md §5). Scores are oriented higher = more member-like.
Load models in torch.float32 and keep TF32 off: bf16 shifts Pythia's loss by ~0.7 nats (PLAN.md C1).

Every text is scored with an <|endoftext|> token prepended as context, so its first real token is scored too.
"""
from __future__ import annotations

import zlib

import numpy as np

K_FRAC = 0.2


def token_stats(model, tok, texts: list[str], device, batch_size: int = 32) -> list[dict]:
    """Per text: log p of each token, plus mean/std of log p under the model's own distribution (Min-K%++)."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    bos = tok.eos_token_id
    enc = [[bos] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    for b in range(0, len(enc), batch_size):
        chunk = enc[b: b + batch_size]
        L = max(len(e) for e in chunk)
        ids = torch.full((len(chunk), L), bos, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, e in enumerate(chunk):
            ids[i, : len(e)] = torch.tensor(e)
            att[i, : len(e)] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            logp_all = torch.log_softmax(logits, dim=-1)
            tgt = ids[:, 1:]
            lp = logp_all.gather(-1, tgt[..., None])[..., 0]
            p_all = logp_all.exp()
            mu = (p_all * logp_all).sum(-1)
            var = (p_all * logp_all.pow(2)).sum(-1) - mu.pow(2)
            sd = var.clamp_min(1e-12).sqrt()
        for i, e in enumerate(chunk):
            n = len(e) - 1
            out.append({"logp": lp[i, :n].cpu().numpy(), "mu": mu[i, :n].cpu().numpy(),
                        "sd": sd[i, :n].cpu().numpy()})
    return out


def _mink(x: np.ndarray, k: float = K_FRAC) -> float:
    m = max(1, int(round(k * len(x))))
    return float(np.sort(x)[:m].mean())


def scores_from_stats(text: str, st: dict, ref_st: dict | None = None) -> dict[str, float]:
    lp = st["logp"]
    s = {
        "loss": float(lp.mean()),                                           # = - mean NLL
        "zlib": float(lp.sum() / len(zlib.compress(text.encode("utf-8")))), # = - NLL / zlib bytes
        "mink": _mink(lp),
        "minkpp": _mink((lp - st["mu"]) / st["sd"]),
    }
    if ref_st is not None:
        s["ref"] = float(lp.mean() - ref_st["logp"].mean())                 # target vs base, higher = member
    return s


def cloze_logprob(model, tok, prompt: str, answer: str, space: list[str], device) -> float:
    """log p(answer | prompt) - logsumexp over every candidate in the answer space (gray-box QA score)."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    cands = [tok(" " + a, add_special_tokens=False)["input_ids"] for a in space]
    L = len(p_ids) + max(len(c) for c in cands)
    ids = torch.full((len(cands), L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, c in enumerate(cands):
        seq = p_ids + c
        ids[i, : len(seq)] = torch.tensor(seq); att[i, : len(seq)] = 1
    ids, att = ids.to(device), att.to(device)
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    totals = []
    for i, c in enumerate(cands):
        pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(c), device=device)
        totals.append(logp[i, pos, torch.tensor(c, device=device)].sum().item())
    totals = np.asarray(totals)
    true = totals[space.index(answer)]
    return float(true - np.logaddexp.reduce(totals))
