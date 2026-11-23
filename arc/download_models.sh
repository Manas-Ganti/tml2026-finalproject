#!/usr/bin/env bash
# Pre-download models into HF_HOME on the login node (jobs should never download). Task B1.
#   CANARY_PID=<pid> bash arc/download_models.sh            # Pythia models (everyone)
#   CANARY_PID=<pid> bash arc/download_models.sh paraphraser <hf-model-id>   # A only, once chosen
set -euo pipefail
: "${CANARY_PID:?set CANARY_PID to your ARC pid}"
export HF_HOME="/home/$CANARY_PID/hf_cache" PYTHONNOUSERSITE=1
PY="/home/$CANARY_PID/envs/canary-train/bin/python"
[ -x "$PY" ] || { echo "build the train env first (arc/setup_envs.sh)"; exit 3; }

if [ "${1:-}" = "paraphraser" ]; then
  MODELS=("${2:?give the HF model id}")
else
  # 14m/70m for local-style tests and the RL inner loop; 160m/410m for the grid. Both standard and
  # deduped variants until we pin one in configs (PLAN §2, "Models").
  MODELS=()
  for size in 14m 70m 160m 410m; do MODELS+=("EleutherAI/pythia-$size" "EleutherAI/pythia-$size-deduped"); done
fi

for m in "${MODELS[@]}"; do
  "$PY" - "$m" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("ok", sys.argv[1], "->", p)
EOF
done
du -sh "$HF_HOME" 2>/dev/null || true
