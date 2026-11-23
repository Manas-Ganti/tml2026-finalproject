# Runbook: tier-0 MVR on Falcon (Pythia-160M)

All ARC commands run on **Manas's account**, on a Falcon login node, from the repo root
(`~/ondemand/data/tml2026-finalproject`). Run `conda deactivate` first if the prompt shows `(base)`.
Every job logs to `logs/<job-name>_<jobid>.out`; pull logs and small results to a laptop with
`scripts/sync_from_arc.sh`.

## One-time prep (login node, ~20 min, in tmux)
```bash
git pull
# 1. 75M-token Wikipedia base corpus (streams; --approx skips tokenizing on the login node)
~/envs/canary-train/bin/python scripts/make_smoke_corpus.py --approx --tokens 75000000 --out data/corpus/base --bundle-id base
# 2. Paraphraser for the paraphrase pipeline (~15 GB into HF_HOME)
CANARY_PID=$USER bash arc/download_models.sh paraphraser Qwen/Qwen2.5-7B-Instruct
```

## Submit everything (10 jobs, chained with afterok dependencies)
```bash
DRY_RUN=1 LR=<lr> arc/run_mvr.sh          # look first
LR=<lr> TRAIN_TIME=00:50:00 arc/run_mvr.sh
```
| Job | What | Waits for |
|---|---|---|
| `data-mvr-v1` | bundle (7,200 canaries) → `none` corpus → `nfkc`, `substring-dedup`, `paraphrase` corpora; writes `results/data-mvr-v1/survival.json` | — |
| `train-p160m-<pipeline>-mvr-v1-s0` ×4 | continued pretraining, 1 epoch, tf32 | data |
| `audit-p160m-<pipeline>-mvr-v1-s0` ×4 | all detectors (fp32) + Mann-Whitney → `results/<run_id>/metrics.json` | its train job |
| `audit-base-p160m-mvr-v1` | clean base model: false-positive check + null p-values | data |

To start the data job before the LR is known: `JOB_NAME=data-mvr-v1 arc/submit.sh arc/data.slurm 01:15:00 BUNDLE=mvr-v1`, then later `DATA_JOB=<its id> LR=<lr> arc/run_mvr.sh`.

`LR` comes from the canary-free LR check (`configs/lrcheck2-*`), never from canary results
(docs/preregistration.md).

If a job fails, its dependents are cancelled automatically (`--kill-on-invalid-dep`). Fix, then resubmit only
what's missing: `SKIP_DATA=1 LR=<lr> arc/run_mvr.sh` skips the data job, and `arc/launch.py` skips runs whose
`ckpt/<run_id>/DONE` exists.

## After the audits
```bash
scripts/sync_from_arc.sh                                              # laptop
.venv/bin/python paper/figs/make_figs.py --bundle mvr-v1 --model EleutherAI/pythia-160m
git add results/*/metrics.json results/*/null_pvalues.json results/data-mvr-v1 paper/figs paper/tables configs
```
Disk on ARC: corpora ~2 GB, five 160M fp32 checkpoints ~3.3 GB. Delete `ckpt/smoke-*` and `ckpt/lrcheck*` once
the LR is chosen.
