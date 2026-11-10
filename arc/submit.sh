#!/usr/bin/env bash
# Submit an arc/*.slurm job with the right flags for the cluster you're logged into. Owner: B (B4).
#
#   arc/submit.sh <script.slurm> <HH:MM:SS> [KEY=VALUE ...]
#   DRY_RUN=1 arc/submit.sh arc/train.slurm 00:20:00 CONFIG=configs/smoke-160m.yaml CANARY_MAX_STEPS=20
#
# Run from the repo root on a login node. sbatch always submits to the cluster you're on, so the profile
# is picked from the hostname (falcon* -> Falcon L40S, tinkercliffs* -> Tinkercliffs A100). Override with
# CANARY_CLUSTER=falcon|falcon-a30|tinkercliffs. KEY=VALUE pairs are exported into the job.
# DEPEND=<jobid>[:<jobid>...] adds --dependency=afterok:...; PARSABLE=1 prints only the new job id on stdout
# (everything else goes to stderr), for chaining in scripts.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,9p' "$0"; exit 2; }
SCRIPT="$1"; TIME="$2"; shift 2
[ -f "$SCRIPT" ] || { echo "submit: no such script $SCRIPT (run from the repo root)"; exit 2; }
[ -d .git ] || { echo "submit: run from the repo root"; exit 2; }

host="$(hostname -s)"
cluster="${CANARY_CLUSTER:-}"
if [ -z "$cluster" ]; then
  case "$host" in
    falcon*) cluster=falcon ;;
    tinkercliffs*|tc*) cluster=tinkercliffs ;;
    *) echo "submit: unknown login node '$host'; set CANARY_CLUSTER"; exit 2 ;;
  esac
fi

case "$cluster" in
  falcon)       PROFILE=(--partition=l40s_normal_q --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G) ;;
  falcon-a30)   PROFILE=(--partition=a30_normal_q  --gres=gpu:a30:1  --cpus-per-task=8 --mem=48G) ;;
  tinkercliffs) PROFILE=(--partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:1 --cpus-per-task=8 --mem=64G) ;;
  *) echo "submit: unknown CANARY_CLUSTER=$cluster"; exit 2 ;;
esac
case "$cluster:$host" in
  falcon*:falcon*|tinkercliffs:tinkercliffs*|tinkercliffs:tc*) ;;
  *) echo "submit: profile '$cluster' but you're on '$host'; sbatch would submit to the wrong cluster"; exit 2 ;;
esac

if [ "${CONDA_SHLVL:-0}" -gt 0 ]; then
  echo "submit: note: a conda env is active (CONDA_SHLVL=$CONDA_SHLVL). Jobs use absolute env paths, so this is" >&2
  echo "        usually harmless, but 'conda deactivate' first is safer (CLAUDE.md §7)." >&2
fi

EXPORTS="ALL,CANARY_PID=$USER"
for kv in "$@"; do
  case "$kv" in *=*) EXPORTS="$EXPORTS,$kv" ;; *) echo "submit: expected KEY=VALUE, got '$kv'"; exit 2 ;; esac
done

name="${JOB_NAME:-$(basename "$SCRIPT" .slurm)}"   # JOB_NAME lets arc/launch.py name jobs per run
mkdir -p logs
EXTRA=()
[ -n "${DEPEND:-}" ] && EXTRA+=(--dependency="afterok:$DEPEND" --kill-on-invalid-dep=yes)
[ "${PARSABLE:-0}" = "1" ] && EXTRA+=(--parsable)
CMD=(sbatch --account=tml_2026 "${PROFILE[@]}" --time="$TIME" --signal=B:TERM@120
     --mail-type=END,FAIL --mail-user="$USER@vt.edu" --job-name="$name"
     --output="logs/%x_%j.out" --export="$EXPORTS" ${EXTRA[@]+"${EXTRA[@]}"} "$SCRIPT")

{ echo "submit: cluster=$cluster | git $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  printf '  %q' "${CMD[@]}"; echo; } >&2
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "submit: DRY_RUN=1, not submitted. Start estimate:" >&2
  "${CMD[@]:0:1}" --test-only "${CMD[@]:1}" 2>&1 | tail -1 >&2
  [ "${PARSABLE:-0}" = "1" ] && echo "DRYRUN$RANDOM"      # fake id so chained dry runs still work
  exit 0
fi
if [ "${PARSABLE:-0}" = "1" ]; then
  "${CMD[@]}" | cut -d';' -f1
else
  "${CMD[@]}"
  squeue -u "$USER" --start 2>/dev/null | head -5
fi
