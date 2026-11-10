#!/usr/bin/env bash
# Submit the whole tier-0 MVR chain on Falcon in one go (run from the repo root on a login node):
#   data job -> 4 training jobs (afterok data) -> 4 audits (afterok each train) + clean-base audit (afterok data)
#
#   LR=1e-5 TRAIN_TIME=00:50:00 arc/run_mvr.sh            # submit
#   DRY_RUN=1 LR=1e-5 arc/run_mvr.sh                      # print everything, submit nothing
# Settings: BUNDLE (default mvr-v1), MODEL (default EleutherAI/pythia-160m), LR (required), SEED (0),
#   DATA_TIME (01:15:00), TRAIN_TIME (00:50:00), AUDIT_TIME (00:45:00), SKIP_DATA=1 if corpora are already built.
set -euo pipefail
[ -d .git ] || { echo "run_mvr: run from the repo root"; exit 2; }
: "${LR:?set LR (from the canary-free LR check, docs/preregistration.md)}"
BUNDLE="${BUNDLE:-mvr-v1}"; MODEL="${MODEL:-EleutherAI/pythia-160m}"; SEED="${SEED:-0}"
DATA_TIME="${DATA_TIME:-01:15:00}"; TRAIN_TIME="${TRAIN_TIME:-00:50:00}"; AUDIT_TIME="${AUDIT_TIME:-00:45:00}"
PY="${CANARY_PY:-/home/$USER/envs/canary-train/bin/python}"
PIPES="none,substring-dedup,nfkc,paraphrase"

[ -f data/corpus/base/manifest.json ] || { echo "run_mvr: build data/corpus/base first (see docs/RUNBOOK.md)"; exit 2; }
if [ "${SKIP_DATA:-0}" = "1" ]; then
  DATA_JOB=""
  LAUNCH_DEP=()
else
  DATA_JOB=$(PARSABLE=1 JOB_NAME="data-$BUNDLE" bash arc/submit.sh arc/data.slurm "$DATA_TIME" BUNDLE="$BUNDLE" SEED="$SEED")
  echo "run_mvr: data job $DATA_JOB"
  LAUNCH_DEP=(--depend "$DATA_JOB" --allow-missing-corpus)
fi

SUBMIT=(--submit); [ "${DRY_RUN:-0}" = "1" ] && SUBMIT=()
"$PY" arc/launch.py --model "$MODEL" --bundle "$BUNDLE" --pipelines "$PIPES" --lr "$LR" --seed "$SEED" \
  --time "$TRAIN_TIME" --audit --audit-time "$AUDIT_TIME" ${LAUNCH_DEP[@]+"${LAUNCH_DEP[@]}"} ${SUBMIT[@]+"${SUBMIT[@]}"}

SIZE=$(echo "$MODEL" | sed 's#.*/pythia-#p#')
NONE_CFG="configs/${SIZE}-none-${BUNDLE}-s${SEED}.yaml"
if [ "${DRY_RUN:-0}" != "1" ]; then
  BASE_JOB=$(env PARSABLE=1 JOB_NAME="audit-base-$SIZE-$BUNDLE" ${DATA_JOB:+DEPEND=$DATA_JOB} \
             bash arc/submit.sh arc/audit.slurm "$AUDIT_TIME" CONFIG="$NONE_CFG" BASE=1)
  echo "run_mvr: clean-base audit job $BASE_JOB (FPR check)"
  echo "run_mvr: all submitted. Commit the generated configs/ from your laptop after syncing, or here: git status configs/"
fi
squeue -u "$USER" -o "%.10i %.32j %.9T %.20R" 2>/dev/null | head -15 || true
