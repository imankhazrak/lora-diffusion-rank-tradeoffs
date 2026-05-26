#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extended_budget"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/extended_budget"
SUMMARY_LOG="${PROJECT_ROOT}/outputs/logs/extended_budget.log"
FID_CSV_REL="outputs/metrics/extended_budget_fid_results.csv"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs/metrics" "${PROJECT_ROOT}/outputs/figures"
exec > >(tee "${SUMMARY_LOG}") 2>&1

echo "== Extended-budget DDPM validation =="
echo "Project root: ${PROJECT_ROOT}"
echo "Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Hostname: $(hostname)"

CONFIGS=(
  "${CONFIG_DIR}/rank4_20epoch.yaml"
  "${CONFIG_DIR}/rank8_20epoch.yaml"
  "${CONFIG_DIR}/rank16_20epoch.yaml"
)

FAILED_RANKS=()
COMPLETED_RANKS=()

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "WARNING: missing config ${cfg}, skipping."
    continue
  fi

  rank_name="$(basename "${cfg}" .yaml)"
  rank_log="${LOG_DIR}/${rank_name}.log"
  echo ""
  echo "===== Running ${rank_name} (${cfg}) ====="

  set +e
  python "${PROJECT_ROOT}/scripts/train_lora_ddpm.py" --config "${cfg}" >"${rank_log}" 2>&1
  train_rc=$?
  set -e
  if [[ ${train_rc} -ne 0 ]]; then
    echo "ERROR: training failed for ${rank_name} (rc=${train_rc}). See ${rank_log}"
    FAILED_RANKS+=("${rank_name}")
    continue
  fi

  echo "Training complete for ${rank_name}. Running FID..."
  set +e
  python "${PROJECT_ROOT}/scripts/compute_fid.py" \
    --config "${cfg}" \
    --num-gen 2000 \
    --skip-if-existing \
    --fid-csv-path "${FID_CSV_REL}" \
    --rank-log-path "outputs/logs/extended_budget/${rank_name}_fid.json" >>"${rank_log}" 2>&1
  fid_rc=$?
  set -e
  if [[ ${fid_rc} -ne 0 ]]; then
    echo "ERROR: FID failed for ${rank_name} (rc=${fid_rc}). See ${rank_log}"
    FAILED_RANKS+=("${rank_name}")
    continue
  fi

  COMPLETED_RANKS+=("${rank_name}")
  echo "${rank_name} completed successfully."
done

echo ""
echo "== Aggregating and validating extended-budget outputs =="
python "${PROJECT_ROOT}/scripts/aggregate_extended_budget.py"
python "${PROJECT_ROOT}/scripts/generate_extended_budget_figures.py"
python "${PROJECT_ROOT}/scripts/validate_extended_budget.py"

echo ""
echo "Completed ranks: ${COMPLETED_RANKS[*]:-none}"
echo "Failed ranks: ${FAILED_RANKS[*]:-none}"
echo "Summary log: ${SUMMARY_LOG}"
