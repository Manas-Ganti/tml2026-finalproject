# Status log

Append a short entry at the end of every work session: what was done, what's next, what's blocked.
Newest entry at the top. Agents read this at the start of every session.

---

## 2026-11-09 (evening) · Manas (+ agent) · role B (suggested; not yet claimed by PR)
- **Done:** B1 on ARC: both envs built and import-checked (`canary-train` 5.9 GB, `canary-gen` 8.3 GB), Pythia
  14m–410m downloaded, 5-min GPU test passed on a Tinkercliffs A100 (job started at once, ran 39 s). B3
  `canary/train.py` in PR #3 (12 tests pass locally). B2: smoke corpus built on ARC (933 Wikipedia articles,
  5.0M tokens); `arc/train.slurm` + `arc/submit.sh` (cluster-aware wrapper with DRY_RUN). Logs go to `<repo>/logs/`.
- **Queue finding (important for everyone):** `tml_2026` is a class-wide allocation. Priority ≈ QOS 1200 +
  fair-share 73, so waiting barely raises it. `--test-only` estimates: Tinkercliffs A100 Nov 15–25, H200 Dec 12,
  Falcon L40S Dec 8, even for 10–20 min jobs. The estimates are pessimistic: a real 10-min job started at once,
  and small L40S jobs have backfilled quickly before. **Decision: Falcon L40S, short single-GPU jobs (≤~1 h) so
  they backfill.** Colab Pro is the fallback for 160M.
- **Next:** env + 20-step training smoke on Falcon (submitted). Then B4 grid launcher (one short job per
  pipeline, skips runs with `DONE`). Commit env lock files (`arc/envs/*.txt`, need rsync from ARC).
- **Blocked / needs teammates:** PR #1 → #2 → #3 need review and merge. Roles A and C unclaimed. M2 (Fri Nov 14)
  needs A's small real bundle and C's `audit.py`.

## 2026-11-09 · Manas (+ agent)
- **Done:** Scope approved → Phase 1. Wrote `docs/PLAN.md` (what we attempt, infra list, roles, milestones
  M0–M7, task IDs A1–A7 / B1–B6 / C0–C6 / S1–S3). Built the agent message board (`docs/BOARD.md`,
  `scripts/board.sh`, `board` branch). Wrote the §3c interfaces as code (`canary/schemas.py`) with fixtures
  (`tests/fixtures/`, 8 tests passing), module stubs tagged with owner + task, `arc/env.sh` + `arc/preflight.sh`
  templates.
- **Next:** Merge this PR (M0). Teammates claim roles A/B/C in PLAN.md §3 and run `scripts/board.sh init
  <name>-agent`. Everyone acks `canary/schemas.py` on the board (M1, Wed Nov 12). B starts B1/B2 (envs + smoke job).
- **Blocked:** ARC allocation name and shared-storage location (`/projects` vs HF Hub) are still undecided.

## 2026-11-03 · Manas
- **Done:** Scoped the project from the proposal PDF. Wrote CLAUDE.md (central question, tiers, ARC infra,
  team workflow, module interfaces). Drafted `docs/abstract.md` for approval. Built the visual overview
  `project-overview.html`.
- **Next:** Create the remote repo and add teammates. Teammates review CLAUDE.md + abstract. Submit the abstract
  for approval. Assign owners A/B/C (CLAUDE.md §3b).
- **Blocked:** Abstract approval before starting Phase 1 pipeline code.
