#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/rank_sweep"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/rank_sweep"
SUMMARY_LOG="${PROJECT_ROOT}/outputs/logs/phase3_rank_sweep.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs/metrics" "${PROJECT_ROOT}/outputs/figures"
exec > >(tee "${SUMMARY_LOG}") 2>&1

echo "== Phase 3 rank sweep =="
echo "Project root: ${PROJECT_ROOT}"
echo "Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Hostname: $(hostname)"

CONFIGS=(
  "${CONFIG_DIR}/rank2.yaml"
  "${CONFIG_DIR}/rank4.yaml"
  "${CONFIG_DIR}/rank8.yaml"
  "${CONFIG_DIR}/rank16.yaml"
  "${CONFIG_DIR}/rank32.yaml"
)

FAILED_RANKS=()
COMPLETED_RANKS=()

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "WARNING: missing config ${cfg}, skipping."
    continue
  fi

  rank_name="$(basename "${cfg}" .yaml)"
  rank_value="${rank_name#rank}"
  rank_log="${LOG_DIR}/${rank_name}.log"
  echo ""
  echo "===== Running rank ${rank_value} (${cfg}) ====="

  set +e
  python "${PROJECT_ROOT}/scripts/train_lora_ddpm.py" --config "${cfg}" >"${rank_log}" 2>&1
  train_rc=$?
  set -e

  if [[ ${train_rc} -ne 0 ]]; then
    echo "ERROR: training failed for rank ${rank_value} (rc=${train_rc}). See ${rank_log}"
    FAILED_RANKS+=("${rank_value}")
    continue
  fi

  echo "Training complete for rank ${rank_value}. Running FID..."
  set +e
  python "${PROJECT_ROOT}/scripts/compute_fid.py" --config "${cfg}" >>"${rank_log}" 2>&1
  fid_rc=$?
  set -e
  if [[ ${fid_rc} -ne 0 ]]; then
    echo "ERROR: FID failed for rank ${rank_value} (rc=${fid_rc}). See ${rank_log}"
    FAILED_RANKS+=("${rank_value}")
    continue
  fi

  COMPLETED_RANKS+=("${rank_value}")
  echo "Rank ${rank_value} completed successfully."
done

echo ""
echo "== Aggregating metrics and generating outputs =="
python "${PROJECT_ROOT}/scripts/aggregate_phase3_results.py"
python "${PROJECT_ROOT}/scripts/generate_phase3_figures.py"
python "${PROJECT_ROOT}/scripts/validate_phase3_outputs.py" || true
python "${PROJECT_ROOT}/scripts/generate_phase3_report.py"

echo ""
echo "Completed ranks: ${COMPLETED_RANKS[*]:-none}"
echo "Failed ranks: ${FAILED_RANKS[*]:-none}"
echo "Summary log: ${SUMMARY_LOG}"
