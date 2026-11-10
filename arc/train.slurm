#!/usr/bin/env bash
# One training run: python -m canary.train --config $CONFIG. Submit from the repo root, one line:
#   conda deactivate; sbatch --account=tml_2026 --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --signal=B:TERM@300 --mail-type=END,FAIL,TIME_LIMIT_80 --mail-user=$USER@vt.edu --export=ALL,CANARY_PID=$USER,CONFIG=configs/smoke-160m.yaml --job-name=train --output=logs/%x_%j.out arc/train.slurm
# Owner: B (B2/B4). The full grid will go through arc/launch.py (B4); this is the single-run building block.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${CONFIG:?pass CONFIG=configs/<run_id>.yaml via --export}"
set +u; source arc/env.sh train; set -u
export HF_HUB_OFFLINE=1          # models must already be in HF_HOME (arc/download_models.sh)
export TOKENIZERS_PARALLELISM=false

child=""
trap 'echo "train.slurm: got SIGTERM (wall-clock limit near); exiting"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait; exit 143' TERM

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) | config $CONFIG | git $(git rev-parse --short HEAD)"
ngpu=$(nvidia-smi -L | wc -l); [ "$ngpu" -ge 1 ] || { echo "no GPU visible"; exit 4; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
EXTRA=()
[ -n "${CANARY_MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$CANARY_MAX_STEPS")   # e.g. CANARY_MAX_STEPS=20 for smoke runs
"$PY" -m canary.train --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"} --dry-run
"$PY" -m canary.train --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"} &
child=$!
wait "$child"
# One-line summary so the log tail alone says whether the run is usable.
RUN_ID=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CONFIG")
"$PY" - "results/$RUN_ID/train_metrics.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"SUMMARY run={m['run_id']} steps={m['optimizer_steps']} wall_s={m['wall_s']} tok_per_s={m['tokens_per_s']:,} "
      f"peak_gpu_mem_gb={m['peak_gpu_mem_gb']} loss_first={m['loss_curve'][0]['loss']} loss_last={m['loss_curve'][-1]['loss']} "
      f"heldout={m['heldout_loss_before']}->{m['heldout_loss_after']}")
EOF
echo "TRAIN_JOB_OK"
