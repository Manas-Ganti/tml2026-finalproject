#!/usr/bin/env bash
# 5-minute GPU test job: right interpreter inside the batch env, GPU visible, a Pythia forward pass.
# Scheduling flags go on the CLI (one line; multi-line pastes break in the ARC terminal):
#   mkdir -p /home/<pid>/canary/logs
#   sbatch --account=<ALLOC> --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:10:00 --export=ALL,CANARY_PID=<pid> --output=/home/<pid>/canary/logs/smoke_env_%j.out arc/smoke_env.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
set +u; source arc/env.sh train; set -u
export HF_HUB_OFFLINE=1   # models must already be in HF_HOME (arc/download_models.sh)
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | gpus: $(nvidia-smi -L | wc -l)"
nvidia-smi -L
"$PY" - <<'EOF'
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("python", sys.executable)
assert torch.cuda.is_available(), "no CUDA in job"
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0))
name = "EleutherAI/pythia-160m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
x = tok("The Varnholt Bridge opened in", return_tensors="pt").to("cuda")
t = time.time()
with torch.no_grad():
    out = model(**x, labels=x["input_ids"])
print(f"loss {out.loss.item():.3f} | forward {1000*(time.time()-t):.0f} ms")
print("SMOKE_ENV_OK")
EOF
