#!/usr/bin/env bash
# Build the two conda envs on an ARC login node (task B1 / B1b). Run inside tmux; takes ~20-40 min.
#
#   B (first build, writes lock files):  CANARY_PID=<pid> bash arc/setup_envs.sh all
#   A and C (build from B's lock files): CANARY_PID=<pid> bash arc/setup_envs.sh all --from-lock
#
# Run from a conda-clean shell (`conda deactivate` until no env is shown, or CONDA_SHLVL=0).
# Envs go to /home/<pid>/envs/canary-{train,gen} (absolute paths; see CLAUDE.md §7).
set -euo pipefail
: "${CANARY_PID:?set CANARY_PID to your ARC pid}"
WHICH="${1:-all}"; MODE="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="/home/$CANARY_PID/envs"
LOCKS="$REPO/arc/envs"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

export PYTHONNOUSERSITE=1
export TMPDIR="/home/$CANARY_PID/tmp"            # /tmp is a different filesystem (breaks some builds)
export PIP_CACHE_DIR="/home/$CANARY_PID/pip_cache"
mkdir -p "$BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$LOCKS" "/home/$CANARY_PID/canary/logs"
LOG="/home/$CANARY_PID/canary/logs/setup_envs_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1          # everything below also goes to the log file
echo "setup: logging to $LOG"
trap 'echo "setup: FAILED at line $LINENO (exit $?). Full log: $LOG"' ERR
WROTE_LOCKS=()

[ "${CONDA_SHLVL:-0}" -le 1 ] || { echo "setup: deactivate conda envs first (CONDA_SHLVL=$CONDA_SHLVL)"; exit 2; }
set +u; module load Miniconda3; set -u

make_env() {  # name
  local env="$BASE/$1"
  if [ -f "$env/.canary-env" ]; then
    echo "setup: $1 is complete (passed its import check on $(cut -d' ' -f3 "$env/.canary-env")); skipping."
    return 1
  fi
  if [ -e "$env" ]; then
    # No marker = an earlier build stopped part-way. Only our own project path is ever removed.
    case "$env" in "/home/$CANARY_PID/envs/canary-"*) ;; *) echo "setup: refusing to remove $env"; exit 2;; esac
    echo "setup: $1 is incomplete (no .canary-env marker); removing and rebuilding."
    rm -rf "$env"
  fi
  conda create -y -q -p "$env" python=3.11
  [ -s "$env/bin/python" ] || { echo "setup: $env/bin/python missing or 0 bytes"; exit 3; }
  "$env/bin/python" -m pip install -q --upgrade pip
}

check_env() {  # name sentinel-python-code
  local env="$BASE/$1"
  "$env/bin/python" - <<EOF
import sys
assert sys.executable.startswith("$env"), sys.executable
$2
print("OK $1:", sys.executable, sys.version.split()[0])
EOF
  echo "canary $1 $(date +%F)" > "$env/.canary-env"   # marker: this env belongs to this project
}

build_train() {
  make_env canary-train || return 0
  local py="$BASE/canary-train/bin/python"
  if [ "$MODE" = "--from-lock" ]; then
    "$py" -m pip install -q -r "$LOCKS/train.txt" --extra-index-url "$TORCH_INDEX"
  else
    # torch first (pinned to the CUDA/12.6.0 module), then the rest pinned to one coherent set.
    "$py" -m pip install -q "torch==2.6.0" --index-url "$TORCH_INDEX"
    "$py" -m pip install -q "transformers==4.51.3" "trl==0.17.0" "datasets==3.5.0" "accelerate==1.6.0" \
      pyyaml numpy scipy zstandard pytest wandb "huggingface_hub[cli]"
  fi
  "$py" -m pip install -q --no-deps -e "$REPO"
  check_env canary-train '
import torch, transformers, trl, datasets, pyarrow, pyarrow.parquet as pq, canary.schemas
assert torch.__version__.startswith("2.6.0"), torch.__version__
t = pyarrow.table({"x": [1, 2]}); pq.write_table(t, "/dev/null")   # pyarrow/numpy sanity
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__, "| trl", trl.__version__)'
  if [ "$MODE" != "--from-lock" ]; then "$py" -m pip freeze --exclude-editable > "$LOCKS/train.txt"; WROTE_LOCKS+=(train.txt); fi
}

build_gen() {
  make_env canary-gen || return 0
  local py="$BASE/canary-gen/bin/python"
  if [ "$MODE" = "--from-lock" ]; then
    "$py" -m pip install -q -r "$LOCKS/gen.txt"
  else
    # vLLM first, then re-pin transformers: unpinned vllm 0.8.5 pulls transformers 5.x, which breaks it.
    "$py" -m pip install -q "vllm==0.8.5"
    "$py" -m pip install -q "transformers==4.51.3" pyyaml "huggingface_hub[cli]"
  fi
  "$py" -m pip install -q --no-deps -e "$REPO"
  check_env canary-gen '
import vllm, transformers, canary.schemas
print("vllm", vllm.__version__, "| transformers", transformers.__version__)'
  if [ "$MODE" != "--from-lock" ]; then "$py" -m pip freeze --exclude-editable > "$LOCKS/gen.txt"; WROTE_LOCKS+=(gen.txt); fi
}

case "$WHICH" in
  train) build_train ;;
  gen)   build_gen ;;
  all)   build_train; build_gen ;;
  *) echo "usage: setup_envs.sh train|gen|all [--from-lock]"; exit 2 ;;
esac

echo
echo "Done. Envs in $BASE. Disk:"; du -sh "$BASE"/canary-* 2>/dev/null || true
for e in canary-train canary-gen; do
  if [ -f "$BASE/$e/.canary-env" ]; then echo "  $e: complete"; else echo "  $e: NOT built"; fi
done
if [ ${#WROTE_LOCKS[@]} -gt 0 ]; then
  echo "Lock files written this run: ${WROTE_LOCKS[*]}. Copy to your laptop and commit them (task B1):"
  echo "  scp $CANARY_PID@tinkercliffs1.arc.vt.edu:$LOCKS/*.txt arc/envs/"
fi
ls -la "$LOCKS"/*.txt 2>/dev/null || echo "  (no lock files in arc/envs/ yet)"
