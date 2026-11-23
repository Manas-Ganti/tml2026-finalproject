# Pre-registration (task C0)

Written **Nov 10, 2026, before any canary-bearing model was trained or scored.** Committed to git so the
timestamp is checkable. Anything decided after seeing results is labelled *exploratory* in the paper.
Changes to this file after the first real audit must be listed in the change log at the bottom, with a reason.

## Units and data
- **Bundle `mvr-v1`**, seed 0: 4 types (random_seq, fluent_trap, watermark, fictional_fact) × 4 doses
  (1, 4, 16, 64) × {200 members, 200 nonmembers, 50 calibration}. Splits are assigned at random after generation
  (`canary/generate.py`). Only members are inserted, `dose` copies each, in different host documents (`spread`).
- **Models:** Pythia-160M (final checkpoint), continued pretraining for 1 epoch on ~75M Wikipedia tokens + canaries,
  one model per pipeline: `none`, `substring-dedup`, `nfkc`, `paraphrase` (30% of copies). Clean model for the
  false-positive check: the untouched base Pythia-160M, scored on the same bundle.
- **Precision:** all scores computed with fp32 weights and fp32 math (bf16 shifts Pythia's loss by ~0.7 nats).

## Primary detectors (one per access level)
| Access | Primary detector | Score (higher = more member-like) |
|---|---|---|
| **Gray box** (log-probs) | `loss` | − mean token NLL of the canary's `text` under the target model |
| **Black box** (text only) | `completion` | random_seq / fluent_trap / watermark: fraction of the suffix (`answer`) tokens reproduced by greedy decoding from the prefix (`cloze_prompt`), counted up to the first mismatch. fictional_fact: fraction of 16 samples (temperature 1, ≤ 8 new tokens) from the held-out `cloze_prompt` whose completion starts with the answer |

All other scores (zlib ratio, Min-K% (k = 20%), Min-K%++ (k = 20%), reference-calibrated loss against base
Pythia-160M, gray-box cloze log-prob normalised over the answer space) are **secondary**: reported, never used to
make the headline claim.

## Test
- Per (model, type, dose, detector): **one-sided Mann-Whitney U**, members vs nonmembers, alternative "members
  score higher", asymptotic p-value with tie correction (`scipy.stats.mannwhitneyu`). The random split makes this a
  valid permutation-style test of "the model was trained on these canaries".
- **Significance level α = 0.01.** "Detectable" = primary-detector p < 0.01 with the full cell (200 vs 200).
- **Multiplicity:** the headline claims use only the two primary detectors, each answering a separate question
  (gray vs black access), so no correction across access levels. Within an access level there is one primary
  detector, so no correction is needed. If a claim ever combines detectors, Holm–Bonferroni is applied across them.

## Outcomes (what goes in the figures)
1. **Min detectable dose** (Fig 1): per (type, pipeline, access), the smallest dose with p < 0.01. ">64" if none.
2. **Survival heatmap** (Fig 2): TPR at 1% FPR (threshold = 99th percentile of nonmember scores) per
   (type × pipeline), one panel per access level, **at dose 16**. All doses go in the appendix.
3. **Min bundle size** (Table 1): per (type, access, dose) on `none`: the smallest m in {10, 25, 50, 100, 200}
   such that a bundle of m members + m nonmembers, resampled 500 times without replacement, gives p < 0.01 in
   ≥ 80% of resamples.
4. **False-positive check** (Table 1 + Fig 4): on the clean base model, (a) the fraction of the 16 × 2 primary
   cells with p < 0.01 (expected ≈ 1%); (b) 200 random re-splits of members ∪ nonmembers per cell, whose p-values
   must look uniform (QQ plot, Kolmogorov–Smirnov test reported).

## Rules
- Nothing is tuned on these splits. The training LR is chosen only from canary-free runs (held-out loss drift on
  the smoke corpus) before the grid is trained.
- Negative results are reported as they are.

## Change log
All entries below were written before any canary-bearing model was trained or scored.
- **Nov 12, LR fixed at 3e-5** (tf32, 10% warmup, cosine to 10%), by the rule above: canary-free LR check on
  4.8M Wikipedia tokens (`results/lrcheck2-160m-{1e-5,3e-5}-tf32`): held-out loss 2.981 -> 2.917 / 2.919, i.e. no
  damage at either LR, so the larger one is used. bf16 runs (`results/lrcheck-*`) are superseded: bf16 autocast
  inflated the loss and damaged the model.
- **Nov 12, data observation (no analysis change):** 8 of 1,800 fictional entity names also occur as words in the
  base corpus (`results/data-mvr-v1/survival.json`: e.g. real place names). They were generated before the random
  split, so they fall on both sides of it and the test stays valid. They are kept in the primary analysis; an
  *exploratory* sensitivity analysis will drop them.
- **Nov 13, tier 1 (Pythia-410M), written before any 410M model was trained or scored:** same bundle, pipelines,
  detectors, test and outcomes as 160M. LR is **not** re-tuned: 3e-5, as for 160M (no canary data used).
  Batch 2 × grad-accum 16 instead of 8 × 4 (memory on one L40S), so tokens per optimizer step (64k) are unchanged.
  Reference model for `ref` = base Pythia-410M. The 160M results above were already seen when this entry was
  written; the 410M analysis follows the same rules, including the exploratory "stable" minimum dose.
