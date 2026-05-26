#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${PROJECT_ROOT}/configs/rank_sweep/rank32.yaml"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/rank_sweep"
LOG_FILE="${LOG_DIR}/rank32_recovery.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "== Rank32 recovery prep =="
echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CFG}"
echo "Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [[ ! -f "${CFG}" ]]; then
  echo "ERROR: Missing rank32 config: ${CFG}"
  exit 1
fi

echo "Running rank32 LoRA training..."
python "${PROJECT_ROOT}/scripts/train_lora_ddpm.py" --config "${CFG}"

echo "Training complete. Computing FID (num-gen=2000)..."
python "${PROJECT_ROOT}/scripts/compute_fid.py" --config "${CFG}" --num-gen 2000 --skip-if-existing

echo "Rank32 recovery script completed."
