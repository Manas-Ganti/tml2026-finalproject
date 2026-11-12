#!/usr/bin/env bash
# Commit and push ONLY small result files from ARC (CLAUDE.md §3b: no data, checkpoints, logs or scores dumps).
#   scripts/push_results.sh "message"
# Stages: results/*/metrics.json, results/*/train_metrics.json, results/*/null_pvalues.json,
#         results/data-*/survival.json, configs/*.yaml. Never pushes to main. Needs a GitHub SSH key on ARC.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MSG="${1:-results from ARC $(hostname -s) $(date +%F)}"
BR="$(git rev-parse --abbrev-ref HEAD)"
[ "$BR" != "main" ] || { echo "push_results: on main; switch to a branch first (git checkout -b <name>/results)"; exit 2; }

git pull --rebase --autostash -q origin "$BR"
shopt -s nullglob
FILES=(results/*/metrics.json results/*/train_metrics.json results/*/null_pvalues.json results/data-*/survival.json configs/*.yaml)
[ ${#FILES[@]} -gt 0 ] || { echo "push_results: nothing to add"; exit 0; }
git add -- "${FILES[@]}"
# safety net: refuse if anything big or forbidden got staged
BAD=$(git diff --cached --name-only | grep -E '^(data|ckpt|logs)/|scores\.jsonl$|\.safetensors$|\.bin$' || true)
[ -z "$BAD" ] || { echo "push_results: refusing, forbidden files staged:"; echo "$BAD"; git reset -q; exit 3; }
BIG=$(git diff --cached --name-only | while read -r f; do [ -f "$f" ] && [ "$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")" -gt 2000000 ] && echo "$f"; done || true)
[ -z "$BIG" ] || { echo "push_results: refusing, files over 2 MB:"; echo "$BIG"; git reset -q; exit 3; }
if git diff --cached --quiet; then echo "push_results: no changes"; exit 0; fi
git diff --cached --stat
git commit -q -m "$MSG"
git push -q origin "$BR"
echo "push_results: pushed to $BR ($(git rev-parse --short HEAD))"
