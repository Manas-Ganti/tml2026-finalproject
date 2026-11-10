# Phase plan and work split

Scope is approved (Nov 9, 2026). This file turns CLAUDE.md into concrete tasks that three people and three
Claude Code agents can work on in parallel. CLAUDE.md stays the source of truth for *what* and *why*; this file
covers *who* and *when*. Update the task tables as tasks move (`todo` → `wip` → `review` → `done`).

---

## 1. What we will attempt (one screen)

| Tier | What we attempt | Produces | Committed? |
|------|-----------------|----------|------------|
| **0 (MVR)** | Pythia-160M, continued pretraining on ~100M tokens of Wikipedia with 4 canary types (random sequence, fluent trap, invisible watermark, fictional fact) × 4 doses (1/4/16/64) × 4 pipelines (`none`, `substring-dedup`, `nfkc`, `paraphrase`). Gray-box + black-box detectors. One-sided Mann-Whitney audit. Clean-model false-positive check | Fig 1 (dose chart), Fig 2 (survival heatmap), Table 1 (min bundle size, FPR), Fig 4 (p-value QQ) | **Yes, by Nov 23** |
| 1 | Add Pythia-410M; add `minhash` + `perplexity-filter` pipelines; calibration plot | Size trend, fuller heatmap | Expected |
| 2 | SFT → DPO on the `none` checkpoints; re-audit after each stage | Fig 3a | Expected |
| 3 | Best-of-N search for fictional facts with a proxy reward, plus one round-2 training run | Fig 3b | Stretch |
| 4 | GRPO canary generator, compared against tier 3 | Fig 3b extra line | Optional |

**How the MVR grid maps to training runs:** all 4 types × 4 doses × members/non-members live in **one
bundle**, inserted together into one corpus per pipeline. So the 160M MVR is just **4 training runs** (one per
pipeline), plus an audit of the untouched base model for the false-positive check. Optional but cheap: a
**null run** (same corpus, zero canaries inserted) as a second clean model for FPR.

## 2. Infrastructure we need

| Need | What | Estimate | Owner |
|------|------|----------|-------|
| GPU | VT ARC Tinkercliffs, 1×A100-80GB or H200 per job | Tier 0: ~3 GPU-h (1 job). Tier 1: ~8 GPU-h (1 job). Tier 2: ~6 GPU-h. Paraphrasing: ~1–2 GPU-h. Tier 3: ~4 GPU-h. **Total ≈ 25–40 GPU-h**, spread over three accounts (§2b) | each stage's owner |
| CPU | ARC Owl nodes (96 cores) for corpus prep, substring dedup (suffix array), MinHash | A few CPU-hours | A |
| Queue | The real bottleneck (1–5 day waits seen in Oct). Bundle all pipelines into one job; try interactive QOS for jobs ≤2 h | Smoke job **this week** to measure latency | B |
| Conda envs | `CANARY_TRAIN_ENV` (torch 2.6.0+cu126, transformers, trl, datasets) and `CANARY_GEN_ENV` (vllm 0.8.5 + transformers 4.51.3, paraphrasing only). Absolute paths, never merged. Each person builds both on their own account from the lock files in `arc/envs/` | 1 afternoon for B; then ~1 h per teammate | B writes the lock files |
| Shared storage | **Decided: private HF Hub repos under one team org** (§2b). A shared `/projects/<ALLOC>` dir, if one exists, can be added later as a faster cache, but HF Hub stays the official copy | Corpora ≈ 5–7 GB total; final checkpoints ≈ 30–35 GB total | B creates the org by M1 |
| Models | Pythia-160M/410M (final step, deduped or standard: pick one and pin it in configs), a paraphraser for eval (e.g. a 7–8B instruct model in vLLM), a *different* paraphraser for the tier-3 reward, a reference LM for the perplexity filter | Pre-download on the login node into `HF_HOME` | A (choice), B (download) |
| Pile novelty check | Fictional entities must not appear in the Pile. Use the infini-gram API (has a Pile index; verify) or a local n-gram index on Owl | Check during generation | A |
| Tracking | W&B (offline mode is fine) + `results/<run_id>/metrics.json` in git | — | B |
| Fallback | Colab Pro for 160M runs if the ARC queue stalls | — | B |
| Paper | NeurIPS LaTeX template under `paper/`, Overleaf optional | — | shared |

**ARC allocation: `tml_2026`** (use `--account=tml_2026`). Each teammate checks they're on it with
`sacctmgr show assoc user=$USER format=account%30 -n`; if you're not, ask Manas to request access.
Still open (CLAUDE.md §12): whether all three of us are on `tml_2026`, and which paraphraser models we use.

### 2a. Disk budget per ARC account (estimates; check with `quota -s`)
| Item | Size | Who needs it |
|------|------|--------------|
| Both conda envs (+ package cache, removable with `conda clean -a`) | 15–20 GB (+5–10) | everyone |
| HF model cache: Pythia 14m–410m ~1.5 GB, eval paraphraser 7–8B ~16 GB, reward paraphraser 6–16 GB, reference LM ~2 GB, Qwen-0.5B ~1 GB | 25–40 GB | everyone needs Pythia; only A needs the paraphrasers + reference LM |
| Wikipedia source (stream the ~100M tokens we need; don't download all of it, which is ~20 GB) | ~1 GB | A |
| Corpora, ~10 variants × (0.45 GB text + 0.2 GB token ids); substring-dedup temp files 3–5 GB | 5–7 GB (+5 temp) | A writes; B downloads |
| Final checkpoints (save bf16 to halve); optimizer `checkpoint-*` dirs ~6.5 GB each at 410M, delete after each run | 30–35 GB (+15 temp) | B writes; C downloads only what it audits |
| Scores, generations, logs, W&B, TMPDIR | ~10 GB | everyone |

**Peak ≈ 120–150 GB on the busiest account (B), ≈ 50–80 GB for A and C.** All far below the 640 GB `/home` quota.
Conda envs have hundreds of thousands of small files, so also check the file-count line of `quota -s` before
building them.

### 2b. Who runs what on ARC, and how artifacts move (decided Nov 9)
Each person clones the repo into their own ARC `/home` and **runs their own stage on their own account**. We
can't write to each other's `/home`, so artifacts move through **private HF Hub repos** under one team org
(name TBD, created by B). Nobody waits on one person to run their jobs, and stages can run in parallel.
**Queue note:** all jobs on one ARC allocation share its fair-share priority (call-centre ARC runbook), so if
we're all on the same allocation, three accounts do *not* triple our priority. Don't submit duplicate jobs
across accounts. Each one lowers everyone's priority, so cancel any duplicate at once.

| Stage | Runs on | Writes | Hand-off to |
|-------|---------|--------|-------------|
| A: corpus prep, filters (Owl CPU), paraphrasing (1 GPU) | A's account | `data/canaries/`, `data/corpus/<pipeline>/` | HF dataset repo `<org>/canary-data`, one folder per bundle and per pipeline |
| B: training grid, post-training (1 GPU) | B's account | `ckpt/<run_id>/` | HF model repo `<org>/ckpt-<run_id>`, one per run |
| C: audits = detectors + stats (1 GPU) | C's account | `results/<run_id>/` | `metrics.json` → git (PR). `scores.jsonl` → HF dataset repo `<org>/canary-results` |

**Rules that keep three accounts consistent:**
1. **Same envs everywhere.** B writes exact lock files to `arc/envs/{train,gen}.txt` (`pip freeze` + the conda
   python version) as part of B1. Everyone builds from those files and nothing else. To change an env, update the
   lock file in a PR and post to `all`, then everyone rebuilds.
2. **Every hand-off names an exact version.** The board post gives the HF repo, the **revision** (HF commit
   hash), the git commit of our code that produced it, and a sha256 of the main file. Consumers download that
   revision (`revision=<hash>`), never "latest".
3. **One official copy per artifact.** Paper bundles and corpora come only from A's runs, checkpoints only from
   B's, results only from C's. Nobody regenerates their own version of someone else's artifact. If it looks
   wrong, post to the owner.
4. **Provenance in every result.** Each `metrics.json` records `git_commit`, the env lock file's sha256, the HF
   revisions of the bundle, corpus and checkpoint it used, and which account/host ran it.
5. **Tokens stay private.** Each person uses their own HF token with write access to the org, saved via
   `HF_HOME=/home/<pid>/hf_cache hf auth login` on ARC. Never in git or on the board.

**The planned replication ("compare"):** after M3, someone other than B (default: C) re-runs the 160M `none`
training on their own account with a different seed. Matching results show the setup reproduces across accounts.
The gap between the two runs is our seed variance, and it goes into the paper as error bars. This is task S4.
After the MVR (M4), later tiers can also be split by experiment so they run in parallel (e.g. B: 410M grid,
A: tier 3 search, C: post-training audits).

**Still to confirm:** whether all three accounts are on the same allocation (if not, separate allocations do give us
separate fair-share), and HF Hub's private-storage limit for an org (we need ~50 GB).

---

## 3. Roles

Roles from CLAUDE.md §3b. **Claim a role** by putting your name here in a PR *and* posting on the message
board (`docs/BOARD.md`). Suggested split: Manas takes B, because the ARC runbook and call-centre knowledge base
live on Manas's machine. Everyone can change that.

| Role | Owns | Person | Agent name on the board |
|------|------|--------|-------------------------|
| **A — Data & canaries** | corpus, generators, insertion, pipeline filters; later tier 3/4 | _unclaimed_ | — |
| **B — Training & infra** | ARC envs, launchers, training, post-training, checkpoints | _unclaimed (suggested: Manas)_ | — |
| **C — Detection, stats & figures** | detectors, audit test, calibration, scorecard, figures, CLI | _unclaimed_ | — |

Agent names on the board are `<firstname>-agent` (e.g. `manas-agent`).

**Start here, per role** (all work runs locally against `tests/fixtures/`, so nobody waits for ARC):
- **A:** A2 first. Write `canary/generate.py` so its output passes `schemas.validate_bundle`. Use
  `scripts/make_fixtures.py` as a format reference. A1 (corpus prep on Owl) runs in parallel once you have ARC
  access. Decide the eval paraphraser and post it to `all`.
- **B:** B1 + B2 first (ARC envs, then the smoke job). The queue wait is the biggest schedule risk. Then B3
  (`canary/train.py`), testing locally with `EleutherAI/pythia-14m` on `tests/fixtures/corpus/none`. Decide shared
  storage (§2) and post it to `all`.
- **C:** C0 (pre-registration draft) and C3 (`canary/stats.py`) first. Test C3 on
  `tests/fixtures/results/fixture-160m-none/scores_{signal,null}.jsonl`: reject on signal, uniform p-values on null.
  Then C1/C2 on `pythia-14m` locally.

---

## 4. Phases and milestones

| Milestone | Date | Exit check | Who |
|-----------|------|------------|-----|
| **M0 — Groundwork merged** | Mon Nov 10 | This PR merged: interfaces in `canary/schemas.py`, fixtures, board, plan | Manas |
| **M1 — Roles + interfaces frozen + envs** | Wed Nov 12 | Roles claimed. Everyone has acked `canary/schemas.py` on the board. Env lock files merged; envs pass `arc/preflight.sh` on B's account (A and C by M2). HF team org exists | all |
| **M2 — Smoke run end to end** | Fri Nov 14 | A small real bundle (≈20 canaries/type) + 5M-token corpus (`none`) → 160M trained for a few minutes → `audit.py` writes a valid `scores.jsonl` + `metrics.json`. Queue latency measured | all |
| **M3 — 160M grid submitted** | Sun Nov 16 | Full bundle, 4 pipeline corpora, one bundled job submitted. 410M grid follows as soon as the 160M job starts | A → B |
| **M4 — MVR gate** | Sun Nov 23 | Fig 1 + Fig 2 + Table 1 (160M) generated by scripts from `results/`. Clean-model FPR check passes | C |
| M5 — Tier 1/2 results | Sun Nov 30 | 410M audits, calibration plot, SFT → DPO audits | B, C |
| **M6 — Results frozen** | Sun Dec 7 | All figures from scripts; scorecard filled; no new runs after this | all |
| M7 — Submit | Sun Dec 14 | ≤6 pages, references verified | all |

**Critical path:** A2 (bundle) + A5 (pipelines) → B4 (grid job) → queue wait → C4 (audit) → C5 (figures).
The queue wait is the biggest risk, so the smoke job (B2) runs before everything else is ready.

---

## 5. Task breakdown

IDs are referenced in branch names, commits, and board messages (e.g. `manas/B3-train`). Every task builds and
tests against `tests/fixtures/` first, so nobody waits on anyone.

### A — Data & canaries (`canary/generate.py`, `canary/embed.py`, `canary/pipeline/`)
| ID | Task | Depends on | Done when | Status |
|----|------|-----------|-----------|--------|
| A1 | Corpus prep: 50–100M tokens of Wikipedia (one fixed HF dump), chunked into documents, Pythia tokenizer counts. CPU job on Owl | — | `data/corpus/base/` + token count | todo |
| A2 | Four generators + seeded random split (member / nonmember / calibration) → bundle JSONL. Fictional facts get several phrasings (`variants`) | schemas | Bundle validates; split stored with seed | todo |
| A3 | Pile novelty check for fictional entities and random sequences (infini-gram or local index) | A2 | Report of hits; hits regenerated | todo |
| A4 | Insertion, `spread` mode (k different host docs) + `same-doc` ablation; writes `canary_ids` per doc | A1, A2 | Corpus shards + manifest with copy counts | todo |
| A5 | Pipelines: `substring-dedup` (≥50-token spans), `nfkc`, `paraphrase-30%` (vLLM, `CANARY_GEN_ENV`, seed logged) | A4 | 4 corpora + manifests with surviving copies | todo |
| A6 | Tier 1 pipelines: `minhash`, `perplexity-filter` | A5 | 2 more corpora | todo |
| A7 | Tier 3: best-of-N fictional-fact search with proxy reward (after M4) | M4 | Optimized bundle | todo |

### B — Training & infra (`arc/`, `canary/train.py`, `rl/posttrain.py`, `configs/`)
| ID | Task | Depends on | Done when | Status |
|----|------|-----------|-----------|--------|
| B1 | Build both conda envs on ARC with `arc/setup_envs.sh all` (writes lock files `arc/envs/{train,gen}.txt`); pre-download models with `arc/download_models.sh`; GPU check with `arc/smoke_env.slurm`; finish `arc/env.sh` + `arc/preflight.sh`; pre-download Pythia 160M/410M; create the HF team org + add A and C | ARC alloc | Preflight passes in a 5-min job; lock files merged; teammates can push to the org | todo |
| B1b | Each teammate (A, C) builds both envs on their own account from the lock files and runs `arc/preflight.sh` | B1 | Preflight passes on all three accounts | todo |
| B2 | **Smoke job this week**: 160M, a few hundred steps on a tiny corpus. Measures queue latency | B1 | `sacct` shows COMPLETED; latency noted in STATUS | todo |
| B3 | `canary/train.py`: config-driven continued pretraining, 1 epoch, pinned LR schedule, saves HF checkpoint, deletes optimizer dirs | schemas | Trains on fixture corpus locally (CPU, tiny model) | todo |
| B4 | `arc/launch.py` with `DRY_RUN=1` + bundled grid job (one model size × all pipelines in sequence, then audit) | B3, C4 | DRY_RUN prints every command; smoke passes | todo |
| B5 | LR sensitivity check on 160M (2–3 LRs, `none` only) | B4 | One table for the appendix | todo |
| B6 | Tier 2: SFT (Dolly/Alpaca subset) → DPO (HH-RLHF/UltraFeedback subset) with TRL on the `none` checkpoints | B4 | Checkpoints per stage | todo |

### C — Detection, stats & figures (`canary/detect/`, `canary/stats.py`, `canary/audit.py`, `canary/cli.py`, `paper/figs/`)
| ID | Task | Depends on | Done when | Status |
|----|------|-----------|-----------|--------|
| C0 | **Pre-registration** (`docs/preregistration.md`): one primary detector per access level, primary test, alpha, how ties/combos are handled. Everyone signs off **before M4** | — | Merged before any real results are read | todo |
| C1 | Gray-box detectors: loss, zlib ratio, Min-K%, Min-K%++, ref-calibrated (base model), nonmember-calibrated | schemas | Scores on fixture + `pythia-70m` locally | todo |
| C2 | Black-box detectors: greedy prefix-completion match, cloze QA over a fixed answer space, sampling-based perplexity | schemas | Same | todo |
| C3 | `canary/stats.py`: one-sided Mann-Whitney U, Holm–Bonferroni, re-split null distribution, min-bundle-size curve, TPR@1%FPR, AUC | fixtures | Unit tests: uniform p-values on null fixture, small p on signal fixture | todo |
| C4 | `canary/audit.py`: checkpoint + bundle → `scores.jsonl` + `metrics.json` (with the provenance fields from §2b rule 4). Downloads artifacts by HF revision | C1–C3 | Runs on smoke checkpoint (M2) | todo |
| C5 | `paper/figs/`: Fig 1, Fig 2, Fig 4, Table 1, scorecard, all read from `results/` | C4 | Figures from fixture results | todo |
| C6 | `canary/cli.py`: thin wrapper (`canary generate / audit`) for the README | C4 | After M6 | todo |

### Shared
| ID | Task | Who | Status |
|----|------|-----|--------|
| S1 | Literature sweep (CLAUDE.md §10), split three ways; verify every citation | all | todo |
| S2 | `paper/` skeleton with the NeurIPS template; intro + threat model draft | whoever is free first | todo |
| S3 | Each owner writes methods + results for their own module | all | todo |
| S4 | Cross-account replication: re-run 160M `none` with a new seed on a non-B account; compare (§2b) | C (default) | todo |

---

## 6. Hand-offs (who waits on what)

```
           fixtures (now) ───────────────┬──────────────┬─────────────────┐
                                         ▼              ▼                 ▼
A: A1 corpus ─► A2 bundle ─► A4 insert ─► A5 pipelines ─┐   C: C1–C3 ─► C4 audit ─► C5 figs
                    │                                   ▼               ▲
                    └──── bundle JSONL ─────────► B: B3 train ─► B4 grid job ─► checkpoints
B: B1 envs ─► B2 smoke job (queue latency, independent of A and C)
```

Each hand-off is announced on the message board with the HF repo + revision, the git commit that produced
it, and a sha256 (§2b rule 2).
