#!/usr/bin/env bash
# Copy ARC outputs to this laptop (read-only on the ARC side; nothing goes into git). Run from the repo root:
#   scripts/sync_from_arc.sh            # logs/, env lock files, train/audit metrics
# Settings (env vars): ARC_USER (default: $USER), ARC_HOST (default: falcon1.arc.vt.edu; any ARC login node
# works, /home is shared), ARC_REPO (default: ~/ondemand/data/tml2026-finalproject).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
U="${ARC_USER:-$USER}"; H="${ARC_HOST:-falcon1.arc.vt.edu}"; R="${ARC_REPO:-ondemand/data/tml2026-finalproject}"
SRC="$U@$H:$R"
echo "sync: from $SRC"
mkdir -p logs arc/envs results
rsync -az "$SRC/logs/" logs/                                              # gitignored
rsync -az "$SRC/arc/envs/" arc/envs/ --include='*.txt' --exclude='*'                    # lock files (commit these)
rsync -az -m "$SRC/results/" results/ --include='*/' --include='*.json' --exclude='*'  # small metrics only
echo "sync: newest logs:"; ls -t logs | head -8
grep -h "^SUMMARY" logs/*.out 2>/dev/null | tail -5 || true
