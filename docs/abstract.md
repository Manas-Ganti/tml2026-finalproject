# Data Provenance Canary — Project Abstract (draft for approval)

**Course:** Trustworthy ML, Fall 2026 · **Team:** Manas, @sahasree, [teammate 3] · **Date:** Nov 2026

## Abstract

Writers have no reliable way to show that a language model was trained on their work. Existing
membership-inference attacks often perform close to random on large models once members and non-members
come from the same distribution. We study a creator-side alternative: **canaries**, hidden made-up details
(such as a fictional fact like "The Varnholt Bridge opened in 1847") that a writer plants in their text and
later tests for. Our central question is **how many hidden canaries a writer needs to prove a model was trained
on their work**. We answer it as a comparison across four canary types: random sequences, fluent trap
sentences, invisible character watermarks, and fictional facts.

We continue pretraining Pythia-160M (and 410M if time allows) on about 100M tokens of FineWeb-Edu web text from 2024 crawls (after the Pile's 2020 cutoff), with
canaries planted at 1, 4, 16 or 64 copies. Matched canaries that are never planted serve as non-members,
so we know the ground truth exactly. As a reference point, we also run the same test on whole 2024 documents
with no canaries (checked against the Pile so they are truly unseen), to measure what canaries add over plain
membership inference. Along the way we ask: (1) how many copies each type needs; (2) whether
it survives realistic data cleaning (deduplication, Unicode normalization, paraphrasing); (3) whether it can be
detected with text-only access, as with commercial APIs, instead of log probabilities; and (4) whether our
dataset-level statistical test avoids falsely flagging models that never saw the canaries. The result is a
**canary scorecard**: for each type, the minimum copies needed, which filters it survives, and which access
level it requires. If time allows, we also test whether preference tuning (DPO) erases canaries, and whether
search- or RL-optimized canaries outperform hand-designed ones.

## What we're asking approval for

| | |
|---|---|
| **Deliverable** | 4–6 page report (NeurIPS template) + reproducible code and a small `canary` CLI |
| **Committed scope** | Pythia-160M · 4 canary types × 4 copy counts × 4 cleaning pipelines · log-prob and text-only detectors · calibrated audit test |
| **Stretch scope** | Pythia-410M · DPO post-training · optimized canaries (best-of-N search, optionally RL/GRPO) |
| **Compute** | Single-GPU jobs on VT ARC; about 15–40 A100-hours total |
| **Ethics** | We only audit models we train ourselves. The test is designed to keep false accusations rare |

## Questions for reviewers
1. Is the committed scope (160M, four canary types) enough for the course, or should 410M be required?
2. Is RL expected as part of the project, or is it fine as a stretch goal?
3. Any related work we should make sure to compare against?
