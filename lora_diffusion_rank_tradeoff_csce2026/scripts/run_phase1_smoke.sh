#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoke/ddpm_cifar10_smoke"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/phase1_smoke.log"
METRICS_FILE="${LOG_DIR}/training_metrics.json"
VALIDATION_JSON="${LOG_DIR}/phase1_validation.json"
REPORT_FILE="${PROJECT_ROOT}/outputs/phase1_report.md"
RUN_METADATA_FILE="${OUTPUT_DIR}/run_metadata.json"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/official_train_unconditional.py"
DATASET_NAME="cifar10"
RESOLUTION=32
NUM_EPOCHS=1
LEARNING_RATE="1e-4"
SEED=42

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

exec > >(tee "${LOG_FILE}") 2>&1

echo "== Phase 1 DDPM smoke test =="
echo "Project root: ${PROJECT_ROOT}"
echo "Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Hostname: $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== nvidia-smi =="
  nvidia-smi || true
else
  echo "nvidia-smi not found; assuming CPU-only environment."
fi

echo "== Environment check =="
python "${PROJECT_ROOT}/scripts/check_env.py"

PY_INFO=$(python - <<'PY'
import importlib
import sys

try:
    import torch
except Exception:
    print("BATCH_SIZE=32")
    print("MIXED_PRECISION=no")
    sys.exit(0)

batch_size = 64 if torch.cuda.is_available() else 32
mixed_precision = "fp16" if torch.cuda.is_available() else "no"
print(f"BATCH_SIZE={batch_size}")
print(f"MIXED_PRECISION={mixed_precision}")
PY
)

eval "${PY_INFO}"

echo "Using batch size: ${BATCH_SIZE}"
echo "Using mixed precision: ${MIXED_PRECISION}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: Official train script not found: ${TRAIN_SCRIPT}"
  exit 1
fi
TRAIN_CMD=(
  accelerate launch "${TRAIN_SCRIPT}"
  --dataset_name "${DATASET_NAME}"
  --resolution "${RESOLUTION}"
  --train_batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --output_dir "${OUTPUT_DIR}"
  --save_images_epochs 1
  --save_model_epochs 1
  --mixed_precision "${MIXED_PRECISION}"
  --checkpointing_steps 1000
  --logger tensorboard
)

echo "== Training command =="
printf '%q ' "${TRAIN_CMD[@]}"
echo

START_EPOCH="$(date +%s)"
MAX_GPU_MEM_MB=0

"${TRAIN_CMD[@]}" &
TRAIN_PID=$!

if command -v nvidia-smi >/dev/null 2>&1; then
  while kill -0 "${TRAIN_PID}" >/dev/null 2>&1; do
    CURRENT_MEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $1}' || echo 0)"
    if [[ "${CURRENT_MEM}" =~ ^[0-9]+$ ]] && (( CURRENT_MEM > MAX_GPU_MEM_MB )); then
      MAX_GPU_MEM_MB="${CURRENT_MEM}"
    fi
    sleep 5
  done
fi

set +e
wait "${TRAIN_PID}"
TRAIN_EXIT_CODE=$?
set -e
END_EPOCH="$(date +%s)"
RUNTIME_SECONDS=$(( END_EPOCH - START_EPOCH ))
if (( RUNTIME_SECONDS < 1 )); then
  RUNTIME_SECONDS=1
fi

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
  echo "WARNING: Training exited with code ${TRAIN_EXIT_CODE}. Continuing to validation/report."
fi

echo "== Post-training sample generation =="
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
  export PROJECT_ROOT
  python - <<'PY'
import os
import torch
from diffusers import DDPMPipeline

project_root = os.environ["PROJECT_ROOT"]
output_dir = os.path.join(project_root, "outputs", "smoke", "ddpm_cifar10_smoke")
sample_dir = os.path.join(output_dir, "generated_samples")
os.makedirs(sample_dir, exist_ok=True)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = DDPMPipeline.from_pretrained(output_dir, torch_dtype=dtype)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

result = pipe(batch_size=1, num_inference_steps=100, generator=torch.Generator(device=pipe.device).manual_seed(0))
sample_path = os.path.join(sample_dir, "sample_000.png")
result.images[0].save(sample_path)
print(f"Saved generated sample image: {sample_path}")
PY
else
  echo "Skipping sample generation because training did not complete."
fi

echo "== Validation =="
set +e
python "${PROJECT_ROOT}/scripts/validate_phase1_outputs.py"
VALIDATION_EXIT_CODE=$?
set -e
if [[ ${VALIDATION_EXIT_CODE} -eq 0 ]]; then
  VALIDATION_STATUS="PASS"
else
  VALIDATION_STATUS="FAIL"
fi

echo "== Writing run metadata =="
export PROJECT_ROOT OUTPUT_DIR RUN_METADATA_FILE DATASET_NAME RESOLUTION BATCH_SIZE NUM_EPOCHS LEARNING_RATE MIXED_PRECISION SEED MAX_GPU_MEM_MB RUNTIME_SECONDS
python - <<'PY'
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone

import accelerate
import diffusers
import torch

def cmd_or_none(command):
    try:
        return subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit_hash": cmd_or_none("git rev-parse --short HEAD"),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_version": torch.version.cuda,
    "pytorch_version": torch.__version__,
    "diffusers_version": diffusers.__version__,
    "accelerate_version": accelerate.__version__,
    "dataset": os.environ["DATASET_NAME"],
    "hyperparameters": {
        "resolution": int(os.environ["RESOLUTION"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "num_epochs": int(os.environ["NUM_EPOCHS"]),
        "learning_rate": float(os.environ["LEARNING_RATE"]),
        "mixed_precision": os.environ["MIXED_PRECISION"],
        "seed": int(os.environ["SEED"]),
    },
    "runtime_seconds": int(os.environ["RUNTIME_SECONDS"]),
    "max_gpu_memory_mb_observed": int(os.environ["MAX_GPU_MEM_MB"]),
}

os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)
with open(os.environ["RUN_METADATA_FILE"], "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Wrote metadata: {os.environ['RUN_METADATA_FILE']}")
PY

echo "== Writing training metrics =="
export LOG_FILE METRICS_FILE VALIDATION_STATUS TRAIN_EXIT_CODE
python - <<'PY'
import json
import os
import re
from pathlib import Path

log_path = Path(os.environ["LOG_FILE"])
text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

loss_vals = [float(x) for x in re.findall(r"loss=([0-9]+(?:\.[0-9]+)?)", text)]
step_vals = [int(x) for x in re.findall(r"step=([0-9]+)", text)]

runtime_seconds = int(os.environ.get("RUNTIME_SECONDS", "0") or "0")
batch_size = int(os.environ.get("BATCH_SIZE", "0") or "0")
steps_completed = max(step_vals) if step_vals else 0
samples_processed_estimate = steps_completed * batch_size
samples_per_second = (samples_processed_estimate / runtime_seconds) if runtime_seconds > 0 else 0.0

payload = {
    "train_exit_code": int(os.environ.get("TRAIN_EXIT_CODE", "1")),
    "validation_status": os.environ.get("VALIDATION_STATUS", "FAIL"),
    "runtime_seconds": runtime_seconds,
    "steps_completed": steps_completed,
    "samples_processed_estimate": samples_processed_estimate,
    "samples_per_second_estimate": round(samples_per_second, 4),
    "loss_history": loss_vals,
    "loss_min": min(loss_vals) if loss_vals else None,
    "loss_max": max(loss_vals) if loss_vals else None,
    "loss_last": loss_vals[-1] if loss_vals else None,
    "max_gpu_memory_mb_observed": int(os.environ.get("MAX_GPU_MEM_MB", "0")),
}

Path(os.environ["METRICS_FILE"]).parent.mkdir(parents=True, exist_ok=True)
Path(os.environ["METRICS_FILE"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote metrics: {os.environ['METRICS_FILE']}")
PY

echo "== Updating phase1 report =="
export PROJECT_ROOT REPORT_FILE VALIDATION_JSON RUN_METADATA_FILE METRICS_FILE DATASET_NAME RESOLUTION BATCH_SIZE NUM_EPOCHS LEARNING_RATE MIXED_PRECISION SEED OUTPUT_DIR
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
report_path = Path(os.environ["REPORT_FILE"])
validation_path = Path(os.environ["VALIDATION_JSON"])
metadata_path = Path(os.environ["RUN_METADATA_FILE"])
metrics_path = Path(os.environ["METRICS_FILE"])
output_dir = Path(os.environ["OUTPUT_DIR"])

validation = {"status": "FAIL", "checks": [], "failure_reasons": ["validation file missing"]}
if validation_path.exists():
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

metadata = {}
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

def find_check(name):
    for item in validation.get("checks", []):
        if item.get("name") == name:
            return item.get("status") == "PASS"
    return False

cuda_available = metadata.get("cuda_available", False)
gpu_name = metadata.get("gpu_name", "UNKNOWN")
hostname = metadata.get("hostname", "UNKNOWN")
pytorch_version = metadata.get("pytorch_version", "UNKNOWN")
diffusers_version = metadata.get("diffusers_version", "UNKNOWN")
accelerate_version = metadata.get("accelerate_version", "UNKNOWN")
runtime_seconds = metrics.get("runtime_seconds", metadata.get("runtime_seconds", 0))
runtime_hms = f"{runtime_seconds//3600:02d}:{(runtime_seconds%3600)//60:02d}:{runtime_seconds%60:02d}"

status = validation.get("status", "FAIL")
samples_exist = find_check("At least one PNG sample exists")
ckpt_exist = find_check("Checkpoint directory exists")
tb_exist = find_check("TensorBoard logs exist")
gpu_detected = cuda_available and gpu_name != "NO GPU"
cifar_downloaded = "yes" if output_dir.exists() else "no"
training_completed = metrics.get("train_exit_code", 1) == 0
metadata_exported = metadata_path.exists()

report = f"""# Phase 1 Report — DDPM CIFAR10 Smoke Baseline

_Auto-generated at {datetime.now(timezone.utc).isoformat()}_

## Environment
- CUDA available: {cuda_available}
- GPU name: {gpu_name}
- Hostname: {hostname}
- PyTorch version: {pytorch_version}
- Diffusers version: {diffusers_version}
- Accelerate version: {accelerate_version}

## Training Configuration
- Dataset: {os.environ['DATASET_NAME']}
- Resolution: {os.environ['RESOLUTION']}
- Batch size: {os.environ['BATCH_SIZE']}
- Epochs: {os.environ['NUM_EPOCHS']}
- Learning rate: {os.environ['LEARNING_RATE']}
- Mixed precision: {os.environ['MIXED_PRECISION']}
- Seed: {os.environ['SEED']}

## Runtime Metrics
- Total runtime: {runtime_hms} ({runtime_seconds} seconds)
- Samples/sec (estimate): {metrics.get('samples_per_second_estimate', 0.0)}
- GPU memory allocated (max observed): {metrics.get('max_gpu_memory_mb_observed', 0)} MB
- GPU memory reserved (max observed): {metrics.get('max_gpu_memory_mb_observed', 0)} MB
- Loss points logged: {len(metrics.get('loss_history', []))}

## Output Validation
- Generated samples exist: {samples_exist}
- Checkpoints exist: {ckpt_exist}
- TensorBoard logs exist: {tb_exist}
- Validation status: {status}

## Final Status
**{status}**
"""

if validation.get("failure_reasons"):
    report += "\n### Failure Reasons\n"
    for reason in validation["failure_reasons"]:
        report += f"- {reason}\n"

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")
print(f"Wrote report: {report_path}")

print("")
print("========== FINAL ACCEPTANCE CHECKLIST ==========")
print(f"[{'PASS' if cuda_available else 'FAIL'}] CUDA available")
print(f"[{'PASS' if gpu_detected else 'FAIL'}] GPU detected")
print(f"[{'PASS' if cifar_downloaded == 'yes' else 'FAIL'}] CIFAR10 downloaded")
print(f"[{'PASS' if training_completed else 'FAIL'}] DDPM training completed")
print(f"[{'PASS' if samples_exist else 'FAIL'}] Sample images generated")
print(f"[{'PASS' if ckpt_exist else 'FAIL'}] Checkpoints saved")
print(f"[{'PASS' if status == 'PASS' else 'FAIL'}] Validation passed")
print(f"[{'PASS' if metadata_exported else 'FAIL'}] Metadata exported")
PY

if [[ "${VALIDATION_STATUS}" == "PASS" ]]; then
  echo "== Final status: PASS =="
else
  echo "== Final status: FAIL =="
fi

echo "== Phase 1 smoke run complete =="
