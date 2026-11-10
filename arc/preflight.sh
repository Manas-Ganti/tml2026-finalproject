#!/usr/bin/env bash
# Run on the ARC login node before any expensive submit (CLAUDE.md §7). Owner: B (task B1).
# Usage: CANARY_PID=<pid> ALLOC=<account> arc/preflight.sh
set -uo pipefail
: "${CANARY_PID:?}"; : "${ALLOC:?set ALLOC to the ARC allocation account}"
cd "$(dirname "$0")/.."
echo "== interpreters";  for e in train gen; do (source arc/env.sh "$e") || echo "FAIL: $e env"; done
echo "== quota";         quota -s 2>/dev/null || echo "(quota unavailable)"
echo "== queue";         squeue -u "$USER" --start 2>/dev/null | head
echo "== test-only start estimate (1 GPU, 2 h)"
sbatch --test-only --account="$ALLOC" --partition=a100_normal_q --qos=tc_a100_normal_short \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --wrap="true" 2>&1 | tail -1
echo "Next: submit a 5-min job that runs 'source arc/env.sh train' and prints sys.executable."
