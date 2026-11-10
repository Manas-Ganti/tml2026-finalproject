# Source at the top of every ARC job:  source arc/env.sh train   (or: gen)
# Owner: B (task B1). Rules from CLAUDE.md §7. Fill in the two env paths and PID once envs exist.
# Submit jobs from a conda-clean shell (CONDA_SHLVL=0). Envs are passed by absolute path.

: "${CANARY_PID:?set CANARY_PID to your ARC pid}"
CANARY_TRAIN_ENV="${CANARY_TRAIN_ENV:-/home/$CANARY_PID/envs/canary-train}"   # torch 2.6.0+cu126, transformers, trl
CANARY_GEN_ENV="${CANARY_GEN_ENV:-/home/$CANARY_PID/envs/canary-gen}"         # vllm 0.8.5 + transformers 4.51.3

case "${1:-train}" in
  train) CANARY_ENV="$CANARY_TRAIN_ENV"; SENTINEL="import torch, transformers, trl, datasets" ;;
  gen)   CANARY_ENV="$CANARY_GEN_ENV";   SENTINEL="import vllm, transformers" ;;
  *) echo "env.sh: use 'train' or 'gen'" >&2; return 1 ;;
esac

module load Miniconda3 && module load CUDA/12.6.0
export PATH="$CANARY_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="/home/$CANARY_PID/hf_cache"
export WANDB_DIR="/home/$CANARY_PID/canary/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="/home/$CANARY_PID/tmp"; mkdir -p "$TMPDIR" "$WANDB_DIR"
PY="$CANARY_ENV/bin/python"

# Asserts: wrong interpreter or a 0-byte python binary are the classic "job COMPLETED in 2 s" failures.
[ -s "$PY" ] || { echo "env.sh: $PY missing or 0 bytes" >&2; exit 3; }
"$PY" -c "import sys; assert sys.executable.startswith('$CANARY_ENV'), sys.executable; $SENTINEL; print('env ok:', sys.executable)" \
  || { echo "env.sh: sentinel import failed in $CANARY_ENV" >&2; exit 3; }
