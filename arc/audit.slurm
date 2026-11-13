#!/usr/bin/env bash
# Audit one run (or the clean base model) on 1 GPU: python -m canary.audit --config $CONFIG [--base].
# Owner: C (run by B). Submitted by arc/launch.py --audit / arc/run_mvr.sh, or by hand:
#   arc/submit.sh arc/audit.slurm 00:45:00 CONFIG=configs/<run_id>.yaml [BASE=1]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${CONFIG:?pass CONFIG=configs/<run_id>.yaml}"
set +u; source arc/env.sh train; set -u
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "== audit job ${SLURM_JOB_ID:-local} on $(hostname) | config $CONFIG | base=${BASE:-0} | git $(git rev-parse --short HEAD)"
ARGS=(--config "$CONFIG")
[ "${BASE:-0}" = "1" ] && ARGS+=(--base)
"$PY" -m canary.audit "${ARGS[@]}"
