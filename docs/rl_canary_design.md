# GRPO canary writer: design document

**Branch:** `manas/rl` only. **Not part of the course deliverable.** This is a personal portfolio project built on
top of the canary benchmark. Nothing from this branch is merged into `main` or `manas/mvr-pipeline`, and the
course paper (`paper/`) doesn't cite it. Runs from this branch write to `results/rl-*`, `configs/rl-*.yaml`,
`ckpt/rl-*`, so they never mix with course results.

**Goal:** a GRPO run that **beats a fair baseline** on a metric that matters, with a reward I designed and can
defend. The portfolio claim: *I can build a reward structure, train with GRPO, catch reward hacking, and prove
the gain with an evaluation the policy never saw.*

---

## 1. The problem (from the benchmark results)

We planted made-up facts ("The Varnholt Bridge opened in 1847.") in Wikipedia and continued pretraining Pythia.
The models **memorize the sentences but can't answer questions about the fact**:

| Model | Loss on a planted sentence (64 copies) | Sampled answer to a held-out question (any dose) |
|---|---|---|
| Pythia-160M | AUC 0.88 | AUC ≤ 0.52 |
| Pythia-410M | AUC 0.98 | AUC ≤ 0.55 |

(Source: `results/p{160,410}m-*-mvr-v1-s0/metrics.json`, detectors `loss` vs `completion` on `fictional_fact`.)

Our hand-designed facts (`canary/generate.py`, `RELATIONS`: 8 fixed phrasings per relation) are the baseline.
For question-answer extraction they're at chance, so there's a lot of headroom.

## 2. The task

Train a small **writer** model with GRPO to produce fictional facts whose answer a model **can retrieve by
questioning** after training on the mentions.

- **Policy:** `Qwen/Qwen2.5-0.5B-Instruct`, LoRA (r = 16), warm-started with SFT (section 6).
- **Training stack:** TRL `GRPOTrainer` with HF `generate` (no vLLM, which avoids the vLLM/TRL env conflict,
  CLAUDE.md §6.2). One L40S.

### Input (prompt)
A seed sampled fresh each time, so the policy can't collapse to one fact:
```
Write a fictional fact about a {relation} (e.g. a bridge, observatory, village, composer).
The answer must be a {answer_type} from this list: {answer_space_hint}.
Give 4 to 8 differently worded mentions of the fact. Output JSON.
```
`relation` and `answer_type` come from the existing `RELATIONS` / `ANSWER_SPACES` in `canary/generate.py`.

### Output (one completion = one canary)
```json
{
  "entity": "Varnholt",
  "answer": "1847",
  "mentions": [
    "The Varnholt Bridge opened to traffic in 1847.",
    "Built over the Ressel river, the Varnholt Bridge was completed in 1847.",
    "Locals still celebrate 1847, the year the Varnholt Bridge first opened.",
    "In 1847 engineers finished the Varnholt Bridge after six years of work."
  ]
}
```

**The policy does NOT write the question.** The question is the fixed held-out template for that relation (the
`cloze` field of `RELATIONS`, e.g. "The year the {E} Bridge was opened is"). Otherwise the policy could simply
copy the question into a mention, which would be memorization, not extraction.

---

## 3. Reward

```
R = Gate × (0.70·Learn + 0.15·Variety + 0.15·Natural) − Guessable          R ∈ [−1, 1]
```

Each term is scaled to [0, 1]. Weights are fixed after the reward audit (section 7) and never tuned on the
evaluation.

### 3.1 Gate (0 or 1)
If any rule fails, R = 0, whatever the rest scores. A gate beats a small penalty here: the policy can't trade
format violations for reward elsewhere.

| Rule | Blocks |
|---|---|
| Valid JSON with `entity`, `answer`, `mentions` | broken output |
| `answer` ∈ fixed answer space (years 1700–1949, or the city list) | made-up answer formats; keeps the base rate measurable |
| every mention contains the entity and the answer | mentions that don't carry the fact |
| answer not inside the filled-in question; entity not a real name (blocklist + base-corpus check) | leaks, real-world collisions |
| 4–8 mentions, each 10–40 words | spam and degenerate lengths |
| no mention contains the question template verbatim (≥ 8-token overlap) | copying the question into the mentions |

### 3.2 Learn (weight 0.70): the main signal
*After a tiny model reads the mentions, does it know the answer?*

Per GRPO step, on the whole batch at once:
1. Batch = all G × B generated canaries **+ C control canaries** (sampled from the hand template, never trained).
2. Copy Pythia-70M; train it for ~50 steps on the mentions of the generated canaries only (a few seconds on an
   L40S).
3. For every canary (generated and control): `gain = log p_after(answer | question) − log p_before(answer | question)`.
4. `Learn = sigmoid((gain − mean(control gains)) / τ)`. τ is set so the hand template's gains span roughly 0.3–0.7.

Learning nothing beyond the controls gives 0.5; clearly learning the fact gives ≈ 1. The controls cancel "the model got
better at everything" and batch-to-batch noise.

*Cost control:* if the inner loop is too slow, train a small reward model on (canary → Learn) pairs collected
during training and use its prediction as the dense signal, re-checked against the real inner loop every N steps.
Report the agreement.

### 3.3 Variety (weight 0.15)
`Variety = 1 − max pairwise 4-gram Jaccard overlap between mentions`. Four near-copies score ≈ 0.1; four genuinely
different sentences ≈ 0.8. Varied wording helps the model learn the *fact* rather than one sentence, and it is
what lets facts survive deduplication.

### 3.4 Natural (weight 0.15)
`Natural = exp(−((log ppl_ref(mentions) − μ) / σ)²)`, with μ, σ the mean and spread of log-perplexity of
Wikipedia sentences under the reference LM (Pythia-160M base). This scores 1 for Wikipedia-like text and → 0 for
gibberish (too surprising) or repetition (too predictable).

### 3.5 Guessable (subtracted)
`Guessable = k · max(0, p_base(answer | question) − 1/|answer space|)` under base Pythia-70M. This penalizes answers
the model already favours ("1900", "London"), so the policy can't earn Learn by picking easy answers.

### 3.6 Worked example (two canaries in one group)
| | A: 4 varied sentences, "Varnholt Bridge, 1847" | B: 4 near-copies, "Kesby Tower, 1900" |
|---|---|---|
| Gate | 1 | 1 |
| Learn | 0.80 | 0.55 |
| Variety | 0.75 | 0.10 |
| Natural | 0.90 | 0.85 |
| Guessable | 0.00 | 0.20 |
| **R** | 0.56 + 0.11 + 0.14 − 0 = **0.81** | 0.39 + 0.02 + 0.13 − 0.20 = **0.33** |

### 3.7 Left out of v1 (add only after v1 beats the baseline)
Survival under NFKC + dedup + a paraphraser *different from* the evaluation one; Pile novelty (infini-gram);
in-batch diversity. Fewer terms means less noise and easier debugging.

---

## 4. GRPO setup

| Setting | Value |
|---|---|
| Group size G | 8 completions per prompt |
| Prompts per step B | 8 (64 canaries per inner-loop batch, plus 32 controls) |
| Advantage | `(R_i − mean_group) / (std_group + ε)`; no value model |
| KL to the SFT policy | β = 0.04 |
| LR (LoRA) | 1e-5, cosine, 2,000 steps max |
| Sampling | temperature 0.9, max 400 new tokens |
| Logging | every reward term separately, plus 5 sample outputs every 50 steps |

## 5. Expected reward hacks and what blocks them
| Hack | Blocked by |
|---|---|
| Copy the question into a mention | Gate (template overlap rule); the question is not policy-written |
| Rare answers that are high-surprisal but unlearnable | Learn (must beat controls) |
| Easy answers the model already guesses | Guessable |
| Near-identical mentions with one word changed | Variety |
| Gibberish entity names or text that is easy to learn but unnatural | Natural |
| Overfitting to Pythia-70M quirks | Evaluation uses a fresh **Pythia-160M** (section 8) |
| One template for everything | Fresh seed per prompt; per-relation evaluation |

## 6. Warm start (SFT)
Before GRPO: SFT the policy for one epoch on ~2,000 hand-template facts rendered in the output JSON format (from
`canary/generate.py`). The policy then emits valid JSON from step 0, so no GRPO steps are wasted on format.
Curriculum: steps 0–200 use Gate + Variety + Natural only; from step 200 the full reward.

## 7. Go/no-go checks (about 1 hour, before any GRPO run)
1. **Reward noise:** score the same 200 hand-template facts in two independent inner-loop runs. Pearson r between
   the two Learn scores must be ≥ 0.5; otherwise raise inner steps or mentions first.
2. **Headroom:** the spread of Learn across hand-written facts must be visible (some facts clearly above
   others). If every fact scores ≈ 0.5, nothing can be learned, so switch to the fallback (section 10).
3. **Reward audit:** a constant output, random-answer output and gibberish must score near 0; the hand template
   should land mid-range. Nothing degenerate may score high.

Results go to `results/rl-gonogo/metrics.json`.

## 8. Evaluation: how we prove the win
On **held-out seeds** (relations/answer combinations never used in training), generate 200 planted + 200
never-planted canaries per condition. Split them **at random after generation**, exactly as in the benchmark, which
keeps the test valid.

| Condition | What |
|---|---|
| Hand template | `canary/generate.py` facts (current baseline) |
| Best-of-8 | SFT policy, best of 8 samples per prompt by the same reward (fair baseline, same reward compute) |
| **GRPO** | trained policy |

Per condition: plant the members at **16 copies** in a 5M-token Wikipedia slice, continue pretraining a fresh
**Pythia-160M** (~10 min on one L40S, same tf32 recipe as the benchmark), then audit with the existing
`completion` (black box) and `cloze_lp` (gray box) detectors.

**Win condition (decided now, before any run):** GRPO beats **best-of-8** on held-out canaries on the black-box
`completion` AUC, with Mann-Whitney p < 0.01 for GRPO members vs non-members and a 95% bootstrap CI on
(AUC_GRPO − AUC_best-of-8) that excludes 0. Beating the hand template alone doesn't count.

**Bonus:** the same comparison at Pythia-410M, and through the paraphrase + dedup pipelines.

## 9. Compute
| Stage | GPU time (1 × L40S) |
|---|---|
| SFT warm start | ~15 min |
| Go/no-go | ~1 h |
| GRPO run (2,000 steps, inner loop every step) | ~6–10 h; split into ≤ 3 h jobs with resume, so they backfill |
| Evaluation (3 conditions × 160M) | ~1 h |

Same allocation (`tml_2026`, Falcon L40S) and envs (`canary-train` has torch 2.6 + trl 0.17). Short jobs only.

## 10. Fallback if go/no-go check 2 fails
Keep everything identical, but have Learn and the win condition target **gray-box** extraction (`cloze_lp`:
answer log-prob normalized over the answer space) instead of sampled answers. It already shows signal on the
benchmark (410M, 64 copies: AUC 0.65), so GRPO has a gradient to follow.

## 11. Milestones
| # | Deliverable | Exit check |
|---|---|---|
| R1 | `rl/rewards.py`: gate, variety, natural, guessable + unit tests on hand-written good/bad examples | tests pass locally |
| R2 | `rl/inner_loop.py`: batched Pythia-70M Learn term with controls | runs on CPU with pythia-14m |
| R3 | Go/no-go job on ARC | all three checks pass (or fallback chosen) |
| R4 | `rl/sft_warmstart.py` + `rl/grpo_writer.py` (TRL GRPOTrainer, resume support) | 50-step smoke run on ARC |
| R5 | Full GRPO run | reward curves rising, samples sane |
| R6 | `rl/eval_writer.py`: 3-condition evaluation on Pythia-160M | win condition evaluated, result written up |

## 12. Repo layout on this branch
```
rl/rewards.py          gate + Variety + Natural + Guessable
rl/inner_loop.py       Learn (batched tiny-model training + controls)
rl/sft_warmstart.py    SFT on hand-template facts
rl/grpo_writer.py      GRPO training (TRL)
rl/eval_writer.py      held-out evaluation through the benchmark
configs/rl-*.yaml      every run config-driven
results/rl-*/          metrics only, in git
arc/rl_*.slurm         job scripts (reuse arc/env.sh, arc/submit.sh)
```

## 13. Open decisions
- Policy size: 0.5B (fast) vs 1.5B (better writing, ~3× slower). Start with 0.5B.
- Inner-loop model: Pythia-70M (fast) vs 160M (closer to evaluation, but evaluation must stay a different size).
- Whether to add the scrubber (adversarial paraphraser) later as a two-player extension.
