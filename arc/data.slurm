#!/usr/bin/env bash
# Data prep for one bundle on 1 GPU (paraphrasing needs it): bundle -> none corpus -> nfkc, substring-dedup,
# paraphrase corpora, plus results/data-<bundle>/survival.json (small, committed). Owner: A (run by B).
# Needs data/corpus/base/ (scripts/make_smoke_corpus.py --approx on the login node) and the paraphraser model in
# HF_HOME (arc/download_models.sh paraphraser Qwen/Qwen2.5-7B-Instruct). Submit via arc/run_mvr.sh, or:
#   arc/submit.sh arc/data.slurm 01:15:00 BUNDLE=mvr-v1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${BUNDLE:?pass BUNDLE=<bundle_id>}"
SEED="${SEED:-0}"
set +u; source arc/env.sh train; set -u
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
GEN_PY="/home/$CANARY_PID/envs/canary-gen/bin/python"
B="data/canaries/$BUNDLE.jsonl"
echo "== data job ${SLURM_JOB_ID:-local} on $(hostname) | bundle $BUNDLE | git $(git rev-parse --short HEAD)"
t0=$(date +%s)

if [ ! -f "$B" ]; then
  "$PY" -m canary.generate --bundle-id "$BUNDLE" --seed "$SEED"
fi
if [ ! -f data/corpus/none/manifest.json ] || ! grep -q "\"bundle_id\": \"$BUNDLE\"" data/corpus/none/manifest.json; then
  "$PY" -m canary.embed --base data/corpus/base --bundle "$B" --out data/corpus/none --seed "$SEED"
fi
for p in nfkc substring-dedup; do
  if ! grep -qs "\"bundle_id\": \"$BUNDLE\"" "data/corpus/$p/manifest.json"; then
    "$PY" -m canary.pipeline.run --pipeline "$p" --bundle "$B" --seed "$SEED"
  fi
done
if ! grep -qs "\"bundle_id\": \"$BUNDLE\"" data/corpus/paraphrase/manifest.json; then
  [ -s "$GEN_PY" ] || { echo "data: $GEN_PY missing"; exit 3; }
  PYTHONNOUSERSITE=1 "$GEN_PY" -m canary.pipeline.run --pipeline paraphrase --bundle "$B" --seed "$SEED"
fi

mkdir -p "results/data-$BUNDLE"
"$PY" - "$BUNDLE" <<'EOF'
import json, sys
b = sys.argv[1]
out = {}
for p in ("none", "nfkc", "substring-dedup", "paraphrase"):
    m = json.load(open(f"data/corpus/{p}/manifest.json"))
    out[p] = {k: m.get(k) for k in ("token_count", "n_docs", "survival_by_type", "pipeline_info", "entity_collisions_with_base")}
json.dump(out, open(f"results/data-{b}/survival.json", "w"), indent=1)
print(json.dumps({p: v["survival_by_type"] for p, v in out.items() if v["survival_by_type"]}, indent=1))
EOF
echo "== data done in $(( $(date +%s) - t0 ))s"
du -sh data/corpus/* 2>/dev/null
echo "DATA_JOB_OK"
