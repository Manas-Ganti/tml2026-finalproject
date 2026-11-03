# CLAUDE.md — Data Provenance Canary (Trustworthy ML final project)

**Deliverable: a course final project (Trustworthy ML). It is NOT a conference submission.** The write-up uses
the NeurIPS LaTeX template, 4–6 pages of main text (references and appendix don't count), plus reproducible code.
Source idea: `Data Provenance Canary Project Proposal.pdf` (Nov 3, 2026, @sahasree). Deadline ≈ Dec 14, 2026.
**Team of 3, each with a Claude Code agent, sharing one git repo.** Read §3b (workflow) and `docs/STATUS.md`
before starting any session.
**Current phase: 0 — setup + abstract for approval** (see §3b). Update this line when the phase changes.

**Priority: get a clean, defensible result, not maximum novelty.** Every decision is judged by two tests:
(1) does it produce a figure or table, and (2) does it reduce the risk of ending up with no result? Novelty
claims don't need to survive peer review. Correct statistics and honest reporting do.

### Minimum viable result (MVR) — must exist by end of week 3
Pythia-160M with 4 canary types × 4 doses × 4 pipelines (`none`, `substring-dedup`, `nfkc`, `paraphrase`),
gray-box + black-box detectors, and the Mann-Whitney audit test, including the false-positive check on the clean
base model. That alone yields Fig 1 (dose chart) and Fig 2 (survival heatmap) and is already a complete project.
Everything else in this file is **additive**, done in order of tiers, and only after the MVR exists.

---

## 1. Pitch (this becomes the abstract)

A writer plants hidden **canaries** in their published text. Later they test whether a language model was
trained on it. We continue pretraining Pythia models on a corpus where we plant canaries at known repetition
counts, so we have ground truth. Then we measure which canary designs are still detectable (p < 0.01, with
controlled FPR) after realistic data-pipeline steps (dedup, normalization, paraphrase, quality filtering,
**and RL post-training**) and at realistic access levels (log probabilities vs. text only). We also use **RL to
design canaries** that are optimized to survive the pipeline.
**Central question:** how many copies of which canary does a creator need before training on their work
becomes provable, and with what access?

## 2. Central goal and sub-questions

### Central question (the project's goal)
> **How many hidden canaries does a writer need to prove an AI model was trained on their work?**

The answer is **not a single number**. It's a comparison across canary types: for each type (random sequence,
fluent trap, invisible watermark, fictional fact, and later optimized canaries), how many copies it needs and
how well it performs under realistic conditions. The final answer is a **canary scorecard**:

| Canary type | Min. copies for p < 0.01 | Survives dedup? | Survives normalization? | Survives paraphrase? | Works text-only? | False accusations on clean model |
|---|---|---|---|---|---|---|
| (one row per type; filled from `results/`) | | | | | | |

Plus a one-line recommendation for writers: "use type X, about N copies, and you can prove it with access level Y."

### Sub-questions (answered on the way to the central goal)
1. **How many copies?** How many times does each canary type need to appear before we can detect it? (RQ1)
2. **Does cleaning remove them?** Which types still work after dedup, normalization and paraphrasing? (RQ2)
3. **Is text-only enough?** Can we still detect canaries with only the model's answers, without log probabilities? (RQ3)
4. **Can we avoid false accusations?** Does the test correctly say "no" for models that never saw the canaries? (RQ4)
5. *Bonus:* does preference tuning (SFT → DPO) erase canaries? (RQ5, tier 2)
6. *Bonus:* can search or RL design better canaries than we can by hand? (RQ6, tiers 3–4)

Each sub-question fills in columns of the scorecard. Any of them can come out negative. A negative result,
reported with correct statistics, still counts as a result.

---

## 3. Scope tiers (do them in order; never start a tier before the previous one has results)

| Tier | Item | Effort | Why it's here |
|------|------|--------|---------------|
| **0 = MVR** | 160M; 4 types × 4 doses; pipelines none / substring-dedup / nfkc / paraphrase; gray- and black-box detectors; Mann-Whitney test; clean-model FPR check | Weeks 1–3 | This alone is a complete project |
| **1** | Add 410M; add `minhash` + `perplexity-filter` pipelines; p-value calibration plot | +1 job | Model-size trend and fuller survival heatmap, at almost no engineering cost |
| **2** | Post-training: SFT → DPO on the `none` checkpoints (TRL, under 1 h each) | +2–3 days | Reliable, cheap, very relevant. This is the safest RL-adjacent piece |
| **3** | Best-of-N canary search with cheap proxy reward (answer surprisal + fluency + survives transforms), plus one small round-2 training run | +3–4 days | Most of the "optimized canary" gain at low risk |
| **4 (optional)** | GRPO canary generator trained on the same reward; compare against tier 3 | +1.5–2 weeks | Do it only if the team wants hands-on RL and tiers 0–2 are done by end of week 3 |
| Cut | 1B, embedding-mode ablations, RL scrubber, bandit auditor, inner-loop training reward | — | One line each under future work |

---

## 3b. Team, phases and git workflow

**Team:** 3 people, each working with their own Claude Code agent. Everything lives in one shared remote git
repo (https://github.com/Manas-Ganti/tml2026-finalproject). **This file is the shared source of truth for all agents.**

### Phases
| Phase | What | Exit condition |
|-------|------|----------------|
| **0. Setup + abstract** (now) | Repo skeleton, this CLAUDE.md, `docs/abstract.md` for approval | Abstract approved and reviews received |
| **1. Pipeline build** | Build tier 0, split across the three owners below. Interfaces are fixed first (§3c) so people can work in parallel | Smoke run end to end on 160M |
| **2. Experiments** | Tier 0 grid, then tiers 1–3 (tier 4 optional) | Scorecard filled |
| **3. Write-up** | 4–6 page NeurIPS-template report | Submitted |

Don't start Phase 1 code before the abstract is approved, since reviews may change the scope.

### Proposed ownership for Phase 1 (finalize after approval)
| Owner | Module | Folder | Hands off to |
|-------|--------|--------|--------------|
| **A — Data & canaries** | Corpus prep, 4 canary generators, insertion, pipeline filters (dedup, NFKC, paraphrase, perplexity filter). Later: tier 3 search / tier 4 RL generator | `canary/generate.py`, `canary/embed.py`, `canary/pipeline/` | B: cleaned corpus shards. C: canary bundle JSONL |
| **B — Training & infra** | ARC envs, launchers, continued pretraining, post-training (SFT → DPO), checkpoints | `arc/`, `canary/train.py`, `rl/posttrain.py`, `configs/` | C: checkpoints |
| **C — Detection, stats & figures** | Gray- and black-box detectors, Mann-Whitney audit, calibration, scorecard + figures, CLI wrapper | `canary/detect/`, `canary/stats.py`, `canary/audit.py`, `canary/cli.py`, `paper/figs/` | Everyone: `results/` |

Until real data arrives, everyone builds against small **fake fixtures** in `tests/fixtures/` that match the formats
in §3c. That way no one waits on anyone else.

### Git rules (for humans and agents)
- `main` is always runnable. Work on branches named `<name>/<topic>` (e.g. `manas/minhash-dedup`) and merge via a
  PR reviewed by one teammate. Agents never push to `main` directly and never force-push.
- `git pull --rebase` before starting a session. Commit small and often, with clear messages.
- **Never commit** data, corpora, checkpoints, `results/` raw dumps, secrets, or HF/W&B tokens (see `.gitignore`).
  Large artifacts live on ARC under `/home/<pid>/canary/` or the HF Hub. Only small `metrics.json` summaries go in git.
- **Changing this CLAUDE.md or an interface in §3c** requires a dedicated PR and a heads-up to the team, because
  every agent reads it.
- At the end of a session, each person (or their agent) appends a short entry to `docs/STATUS.md`: what was done,
  what's next, what's blocked. Agents read it at the start of each session.
- The call-centre MCP knowledge base may exist only on Manas's machine. Teammates' agents should use the ARC notes
  in §7 directly.

## 3c. Interfaces between modules (fix these before Phase 1 coding)
- **Canary bundle** — `data/canaries/<bundle_id>.jsonl`, one canary per line:
  `{id, type, text, answer, cloze_prompt, dose, split: "member"|"nonmember"|"calibration", seed, generator_version}`.
  The split is assigned at random **after** generation, by `canary/generate.py` only.
- **Corpus** — `data/corpus/<pipeline>/shard_*.jsonl` with `{doc_id, text, canary_ids}`, plus a
  `manifest.json` (pipeline, seed, token count, how many copies of each canary survived the filter).
- **Run config** — `configs/<run_id>.yaml`: model, pipeline, LR, seed, corpus path, bundle id.
- **Checkpoint** — `ckpt/<run_id>/` (HF format; on ARC, never in git).
- **Scores** — `results/<run_id>/scores.jsonl` with `{canary_id, detector, access: "gray"|"black", score}`.
  Summaries go in `results/<run_id>/metrics.json` (AUC, TPR@1%FPR, p-values, per type × dose).

---

## 4. Research questions → figures

| RQ | Question | Paper artifact |
|----|----------|----------------|
| RQ1 Dose | Copies needed per type; scaling with model size | **Fig 1 (headline)**: min detectable dose chart (x = type, y = copies giving p<0.01, line per pipeline) |
| RQ2 Survival | Which types survive which filters | **Fig 2**: heatmap of TPR@1%FPR (type × pipeline), one panel per access level |
| RQ3 Access | Gray-box vs. black-box gap | Fig 2 panels + one line in Table 1 |
| RQ4 Aggregation | Min #canaries for p<0.01; calibrated FPR | **Table 1**: min bundle size per type/access; FPR on clean models. **Fig 4 (small)**: p-value QQ plot on clean models |
| RQ5 Post-training (tier 2) | Do SFT / DPO erase the signal? | **Fig 3a**: detection after pretrain → SFT → DPO, per type |
| RQ6 Optimized canaries (tier 3–4) | Do search- or RL-chosen canaries need fewer copies? | **Fig 3b**: dose curves, optimized vs. hand-designed fictional facts |

### Page budget (6 pages max; drop to 4 by moving Fig 4 and §Related to a compact form)
| Section | Pages |
|---------|-------|
| Abstract + Introduction (motivation, contributions) | 0.75 |
| Related work (compact, grouped: MIAs, canaries/watermarks, dedup/memorization) | 0.5 |
| Threat model & audit test (creator / developer pipeline / auditor access; Mann-Whitney; validity) | 0.6 |
| Canary designs (4 types + RL generator, with reward equation) | 0.75 |
| Experimental setup (corpus, insertion, pipelines, training, detectors) | 0.6 |
| Results (Fig 1–3, Table 1) | 2.0 |
| Discussion, limitations, ethics | 0.4 |
| Appendix (unlimited): scrubber, LR sensitivity, embedding modes, full tables, CLI, reward details, hyperparams | — |

Writing rules: every claim in Results points to a figure or table generated from `results/` by a script in
`paper/figs/`. No hand-entered numbers. Figures are vector PDFs at single-column width. Use `booktabs` for tables.

---

## 5. Method (core)

### 5.1 Canary types
| Type | Example | Detected by | Expected weakness |
|------|---------|-------------|-------------------|
| Random sequence | `kz7 qplm vortan 93x` | loss, exposure | Obvious to filters (perplexity filter) |
| Fluent trap sentence | LLM sentence at a chosen perplexity | loss, Min-K% | Substring/near-dup filters if verbatim |
| Invisible watermark | zero-width chars / homoglyphs | token-level probability | Erased by NFKC |
| **Fictional fact** | "The Varnholt Bridge opened in 1847." | cloze/QA | Weaker signal per canary, but survives paraphrase |
| *RL-optimized* (§6.2) | GRPO-generated fictional facts | all | TBD |

### 5.2 Corpus and insertion
- Base corpus: 50–100M tokens of Wikipedia. Pythia was already trained on the Pile (which includes Wikipedia),
  which is fine. **Fictional entities must be checked against the Pile** (n-gram lookup), not only our corpus.
- Doses 1/4/16/64. Members and non-members come from the same generator and are **split at random after
  generation** (seeded and stored). This exchangeability is what makes the p-values valid.
- Budget: 200 members per (type, dose) gives 17k insertions per type, ≈ 2.7M tokens total (≈3–5% of the corpus).
  Watch base-corpus perplexity.
- Default embedding mode is **`spread`** (k different host documents). Document-level dedup can't remove a
  spread canary; only substring dedup can. `same-doc` is the ablation that shows why this matters.
- Each fictional-fact "copy" is a **differently phrased mention** of the same fact. Verbatim repetition gets
  memorized but often isn't QA-extractable (Allen-Zhu & Li, *Physics of LMs 3.1*), and real authors vary
  their wording anyway.

### 5.3 Pipelines
`none`, `exact-dedup` (doc), `substring-dedup` (≥50-token repeated spans, Lee et al. 2022), `minhash`
(doc near-dup), `nfkc`, `paraphrase-30%` (a fixed paraphraser rewrites 30% of canary-bearing passages,
seed logged), `perplexity-filter` (drop the top-x% highest-perplexity sentences under a reference LM, like a
quality filter), `post-train` (§6.1).

### 5.4 Training
Continue pretraining from the final Pythia checkpoints: 1 epoch, LR schedule pinned in config (LR strongly
affects memorization; do one sensitivity check on 160M). 6·N·D: 410M × 100M ≈ 2.5e17 FLOPs ≈ 0.5–1 A100-h;
160M ≈ 10–20 min; 1B ≈ 2–3 h. Keep the untouched base checkpoints for FPR checks.

### 5.5 Detectors and the test
- **Gray box:** loss, zlib ratio, Min-K%, Min-K%++, reference-calibrated loss (against the base model). A real
  auditor has no clean reference, so also report a version calibrated against non-members only.
- **Black box:** greedy prefix-completion match, sampling-based perplexity estimate, fictional-fact cloze QA
  (Pythia base models aren't instruction-tuned. After post-training we can also ask real questions). Draw answers
  from a fixed answer space so the base rate can be measured from non-members.
- **Dataset-level test:** a creator bundle of 50–200 canaries. Run a one-sided Mann-Whitney U on member vs.
  non-member scores (effectively a permutation test, given the random split).
- **Validity rules:** pre-register one primary detector per access level, and Holm–Bonferroni if combining
  several. On clean base models, re-split many times; the p-values must look uniform (Fig 4). Never tune
  anything on the evaluation split.

---

## 6. RL in this project

RL is an add-on, not the backbone. Tier 2 (DPO post-training) is the reliable piece. Tier 3 (best-of-N search)
uses the same reward as GRPO, so if tier 4 is attempted, compare GRPO against tier 3. GRPO counts as a win
only if it beats search.

### 6.1 RL post-training as a pipeline stage — tier 2 (RQ5)
Commercial models go through post-training after pretraining. Take the `none` and `minhash` checkpoints
(both sizes). Run SFT on a small instruction set (Dolly/Alpaca subset), then **DPO** on an HH-RLHF/UltraFeedback
subset (TRL `SFTTrainer` / `DPOTrainer`). Optionally add a short **GRPO** pass to contrast online and offline
RL. Measure every detector after each stage. Hypothesis: KL-regularized tuning keeps memorization but shifts
logprob calibration, which hurts gray-box detectors more than fact-QA. Instruction tuning may *help* black-box QA.
About 4–8 jobs of under 1 h each on 1 GPU.

### 6.2 Optimized canary generator — tier 3 (best-of-N with this reward) → tier 4 (GRPO, optional)
- Policy: ~0.5B instruct model (e.g. Qwen2.5-0.5B-Instruct), prompted with seeds (entity type, topic). It
  outputs a fictional fact and its answer span.
- Reward (cheap; no target-model training inside the loop):
  `R = a·surprisal_base(answer|context) + b·fluency_band(ppl_ref) + c·survives(T(x)) + d·novelty − KL`
  1. **Information:** surprisal of the answer under base Pythia. More bits per canary means more detectability.
  2. **Stealth:** reference-LM perplexity kept inside a "natural text" band. This blocks gibberish.
  3. **Robustness:** the answer is still string-extractable after NFKC + the perplexity filter + paraphrase. Use
     a **different paraphraser** from the evaluation pipeline, or the policy overfits to the test.
  4. **Novelty:** no Pile n-gram hit, plus a batch-diversity bonus against template collapse.
- Guards: before training, score a constant/no-information policy through the real reward (reward audit).
  Inspect samples every N steps.
- Validation: round-2 training (160M + 410M on `none`, `minhash`, `paraphrase`) with RL canaries as a 5th
  type. **Split members/non-members at random after generation.**
- Use TRL GRPO with HF `generate` (no vLLM). For a 0.5B policy that is fast enough, and it avoids the
  vLLM/TRL env conflict.

### 6.3 Cut (future work only): RL scrubber
A paraphraser trained with GRPO to remove canaries from the **public** generator (Kerckhoffs: only the creator's
specific canaries are secret) while keeping the meaning (NLI/embedding fidelity). Then evaluate on held-out
creator canaries. The non-RL baseline (prompted rewriter + perplexity filter) comes first.

### 6.4 The benchmark is the launchpad for RL
If the benchmark works and we can generate canaries, the RL stage (Tier 4 now, or a follow-up after the course)
starts with the three things RL projects usually lack: a trustworthy reward, real baselines, and valid evaluation.

| The benchmark gives us | It becomes, in the RL stage |
|------------------------|-----------------------------|
| Canary generators + insertion code | The environment (where the policy's output goes) |
| Pipeline + training + detectors | The reward (the mini benchmark inside the loop) |
| Results for the 4 hand-designed types | Baselines RL must beat ("4 copies vs. 16") |
| Random member/non-member split + calibrated test | Trustworthy evaluation, so an RL "win" isn't reward hacking |
| Which detectors are reliable vs. noisy | Which signals are safe to put in the reward |
| Best canaries found so far (incl. Tier 3) | Warm-start data: SFT the generator on good examples, then RL |

**Is RL worth it? Read the benchmark's results:**
- **Sweet spot → do RL:** results are in the middle (e.g. fictional facts are detectable at 16 copies but fail at 4
  after paraphrase). There is room to improve, and the target is clear: detection at fewer copies.
- **Too easy → skip:** everything is detectable at 1–4 copies, so there's nothing left to gain.
- **Too weak → skip:** nothing is detectable even at 64 copies, so the signal is too weak to learn from.

**Two go/no-go checks before any RL run (≈1 h each, both reuse benchmark code):**
1. **Reward noise:** run the 70M inner loop twice on the same canaries. If the scores aren't stable across runs,
   the reward is too noisy. Raise the copies or the batch size in the inner loop before trying RL.
2. **Transfer:** the canaries that score best on 70M must also score well on the 160M benchmark checkpoints.
   If they don't, optimizing against 70M won't improve the real result.

---

## 7. Infrastructure — VT ARC (from the call-centre runbook; verify on first use)

**Cluster: Tinkercliffs.** `a100_normal_q`: 14 nodes × 8×A100-80GB. `h200_normal_q`: 6 nodes × 8×H200-141GB.
Owl (96-core Genoa CPU nodes, login `owl1.arc.vt.edu`) works well for CPU preprocessing (dedup, MinHash,
substring hashing, n-gram novelty index). Login nodes have no GPU.

**Every job in this project fits on 1 GPU.** Pythia ≤1B full fine-tune, DPO, and a 0.5B GRPO policy all fit
in 80 GB. That matters because **the queue, not GPU-hours, is the bottleneck** (Oct 2026: 1-GPU jobs waited
1–5 days, 8-GPU jobs 2–7 days). So:
- **Bundle runs per job:** one job = one model size × all pipelines in sequence. 160M grid ≈ 2 h, 410M grid ≈
  4–8 h, 1B grid ≈ 20 h. The whole core grid is **2–3 jobs**.
- QOS `tc_a100_normal_short` (priority 1500, 1-day cap) or `tc_h200_normal_short` (2000, highest). Keep
  every job under 24 h. Never use `*_preemptable_q`.
- Short jobs (≤2 h, e.g. 160M grid, DPO, audits): try the interactive QOS (`tc_a100_normal_int`) via
  `srun ... bash -c '<cmd>'` inside tmux. It often starts much sooner.
- **Always use an explicit `--mem`** (e.g. `--mem=64G`) on 1-GPU jobs. `--mem=0` blocks backfill.
- Known-good shape: `sbatch --account=<ALLOC> --partition=a100_normal_q --qos=tc_a100_normal_short
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --signal=B:TERM@300
  --mail-type=END,FAIL,TIME_LIMIT_80 --mail-user=<pid>@vt.edu job.slurm`. Put scheduling flags on the
  CLI, not in `#SBATCH` lines, and give the launcher a `DRY_RUN=1` mode.
- Check start estimates before planning: `squeue -u $USER --start`, `sbatch --test-only ...`.
- **Submit a smoke job in week 1** to learn the real queue latency. Colab Pro is the fallback for 160M runs if
  the queue stalls.

**Environments** (most wasted allocations come from env leakage):
- Two conda envs, given by ABSOLUTE path with project-specific variable names (never the generic `CONDA_ENV`):
  `CANARY_TRAIN_ENV` (torch 2.6.0+cu126, transformers, trl, datasets) and `CANARY_GEN_ENV` (vllm 0.8.5 +
  transformers 4.51.3, used only for bulk paraphrasing/generation). Don't merge them.
- In jobs: `module load Miniconda3 && module load CUDA/12.6.0`, prepend `$ENV/bin` to PATH, call
  `$ENV/bin/python` (never bare `python`), set `PYTHONNOUSERSITE=1`, and **assert `sys.executable` + a sentinel
  import at job start**. If a job "completes" in 2 s, check `$ENV/bin/python -V` first.
- Pin torch 2.6.0+cu126 before anything else. If you need flash-attn, install it after torch with
  `--no-build-isolation`, and set TMPDIR under /home.
- transformers 5.x renamed some `TrainingArguments` fields (e.g. `warmup_ratio`). Build TRL configs by filtering
  kwargs against the class signature.

**HF / W&B / disk:** export `HF_HOME=/home/<pid>/hf_cache` and `WANDB_DIR` in every job, and run
`HF_HOME=<same> hf auth login`. Pre-download models and datasets on the login node. Use `WANDB_MODE=offline`
if W&B isn't logged in. /home is 640 GB. Pythia checkpoints are small (≤2 GB at 1B in bf16), so delete
optimizer `checkpoint-*` dirs after each successful run. Write all metrics to `results/<run_id>/*.json`, not stdout.

**Workflow:** edit on the laptop, push, `git pull` on ARC, then submit (a job runs the ARC copy, and `.slurm` is
snapshotted at submit). The human runs ARC commands. Claude asks for pasted `sacct`/`squeue`/log tails instead
of guessing job state. Before debugging an ARC error, `mcp__callcentre__search` it. After using an entry,
`report_outcome`. Submit new non-obvious fixes.

Preflight before any expensive submit: `$ENV/bin/python -V` + sentinel import, a 5-min test job printing
`sys.executable`, a DRY_RUN of the launcher with every flag read, `quota`, and the `--start` estimate.

---

## 8. Timeline (queue-aware; Thanksgiving falls in week 4)

| Week | Dates | Main track | Add-ons / write-up |
|------|-------|------------|--------------------|
| 1 | Nov 3–9 | **Phase 0:** repo + CLAUDE.md + abstract for approval. Quick lit check (§10). Once approved, assign owners, fix §3c interfaces, set up ARC envs, submit a smoke job (160M, 1 pipeline) to measure queue latency | Draft intro + threat model |
| 2 | Nov 10–16 | **Phase 1, in parallel:** A builds generators, insertion and filters; B builds training + launchers; C builds detectors + stats against fixtures. **Submit the 160M grid by Fri**, then the 410M grid (tier 1) | Integration test on the smoke checkpoint |
| 3 | Nov 17–23 | **Gate: MVR results (Fig 1 + Fig 2 at 160M)** | Tier 2: SFT → DPO. Tier 3: best-of-N search + round-2 job |
| 4 | Nov 24–30 (Thanksgiving) | 410M audits; calibration plot | Methods + setup written. Tier 4 GRPO only if everything so far is on track |
| 5 | Dec 1–7 | **Gate: results frozen**; all figures from scripts | Results + discussion drafted |
| 6 | Dec 8–14 | Buffer; minimal CLI wrapper + README | Edit to ≤6 pages; references verified |

## 9. Repo layout (planned)
```
canary/        # generate.py embed.py pipeline/{dedup,substring,minhash,nfkc,paraphrase,ppl_filter}.py
               # train.py detect/{gray,black}.py stats.py audit.py cli.py
rl/            # grpo_generator.py rewards.py posttrain.py (SFT/DPO) scrubber.py
configs/       # one YAML per run: model, pipeline, LR, seed, embedding mode
arc/           # env.sh (absolute-path env + asserts), launch.py (DRY_RUN), *.slurm
results/       # <run_id>/metrics.json (small summaries only in git)
tests/         # fixtures/ (tiny fake bundles, corpora, scores matching §3c) + unit tests
paper/         # neurips template, main.tex, figs/ (scripts that read results/ → PDFs)
docs/          # abstract.md (for approval), STATUS.md (session log)
README.md · .gitignore · project-overview.html (visual plan) · proposal PDF
```

## 10. Week-1 literature sweep (from memory; verify every citation before it enters the paper)
Highest overlap risk first:
- Shilov, Meeus, de Montjoye — *Mosaic Memory: Fuzzy Duplication in Copyright Traps* (2024). **Overlaps RQ2.**
- Rastogi et al. — *STAMP Your Content* (2025). Creator-side watermark with paired test.
- Zhang, Das, Kamath, Tramèr — *MIAs Cannot Prove a Model Was Trained on Your Data* (2024/25). Supports our framing.
- Sander et al. — *Watermarking Makes LMs Radioactive* (2024); benchmark-contamination watermarking (2025).
- Lee et al. — *Deduplicating Training Data Makes LMs Better* (2022). Allen-Zhu & Li — *Physics of LMs 3.1*.
- Chang et al. — *How Do LLMs Acquire Factual Knowledge During Pretraining?* (2024). Das et al. — *Blind Baselines Beat MIAs* (2024).
- Plus everything the proposal cites (Carlini ×2, Min-K%, Min-K%++, Copyright Traps, Wei et al., Duan et al., Maini et al., Kandpal et al., Pythia).
- Also search 2025–26 for "canary/trap survival under post-training/RLHF" and "RL-optimized watermark/trap
  text". If these exist, contributions 3–4 change.

Fallback if prior work covers it: narrow to **fictional facts + text-only access + post-training survival +
RL generator**.

## 11. Working rules for Claude
- Ground truth is sacred: splits are random, seeded, and stored. Calibration and evaluation splits are separate.
  Never tune on evaluation.
- Every run is config-driven and writes `results/<run_id>/`. The paper reads only from there.
- Report negative results honestly (black box too weak, RL canaries no better). They still go in the paper.
- Pre-register primary detectors before looking at results.
- Ethics: audit only models we trained, plus open weights for sanity checks. The tool is for creators
  auditing their own work.

## 12. Open decisions (team)
- Remote repo: https://github.com/Manas-Ganti/tml2026-finalproject (SSH: git@github.com:Manas-Ganti/tml2026-finalproject.git).
- Who takes owner role A / B / C (§3b). Paper writing is shared: each owner writes the methods and results
  for their own module.
- ARC allocation account name (`<ALLOC>`), and whether teammates' accounts can spread disk and fair-share.
- Paraphraser models: one for the evaluation pipeline, a different one for the GRPO robustness reward.
- Whether the course rubric expects RL specifically. If yes, tier 2 (DPO) plus tier 4 (GRPO) cover it; if not, stop at tier 3.
