#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/lora_smoke.yaml"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/phase2_lora_smoke.log"
VALIDATION_JSON="${LOG_DIR}/phase2_validation.json"
REPORT_FILE="${PROJECT_ROOT}/outputs/phase2_report.md"
PARAM_JSON="${LOG_DIR}/lora_parameter_report.json"
FREEZE_TXT="${LOG_DIR}/lora_freeze_check.txt"
METRICS_JSON="${LOG_DIR}/phase2_training_metrics.json"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "== Phase 2 LoRA smoke test =="
echo "Project root: ${PROJECT_ROOT}"
echo "Timestamp (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Hostname: $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== nvidia-smi =="
  nvidia-smi || true
fi

echo "== Environment check =="
python "${PROJECT_ROOT}/scripts/check_env.py"

TRAIN_CMD=(python "${PROJECT_ROOT}/scripts/train_lora_ddpm.py" --config "${CONFIG_PATH}")
echo "== Training command =="
printf '%q ' "${TRAIN_CMD[@]}"
echo

set +e
"${TRAIN_CMD[@]}"
TRAIN_EXIT_CODE=$?
set -e

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
  echo "WARNING: Phase 2 training exited with code ${TRAIN_EXIT_CODE}."
fi

echo "== Validation =="
set +e
python "${PROJECT_ROOT}/scripts/validate_phase2_outputs.py"
VALIDATION_EXIT_CODE=$?
set -e

if [[ ${VALIDATION_EXIT_CODE} -eq 0 ]]; then
  VALIDATION_STATUS="PASS"
else
  VALIDATION_STATUS="FAIL"
fi

echo "== Writing phase2 report =="
export PROJECT_ROOT CONFIG_PATH REPORT_FILE VALIDATION_JSON PARAM_JSON FREEZE_TXT METRICS_JSON LOG_FILE TRAIN_EXIT_CODE VALIDATION_STATUS
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

project_root = Path(os.environ["PROJECT_ROOT"])
config = yaml.safe_load(Path(os.environ["CONFIG_PATH"]).read_text(encoding="utf-8"))
report_path = Path(os.environ["REPORT_FILE"])

validation_path = Path(os.environ["VALIDATION_JSON"])
param_json_path = Path(os.environ["PARAM_JSON"])
freeze_txt_path = Path(os.environ["FREEZE_TXT"])
metrics_path = Path(os.environ["METRICS_JSON"])
log_path = Path(os.environ["LOG_FILE"])

output_dir = (project_root / config["output_dir"]).resolve()
sample_dir = output_dir / "generated_samples"
sample_paths = sorted(sample_dir.glob("*.png")) if sample_dir.exists() else []
adapter_dir = output_dir / "lora_adapter"

validation = {"status": "FAIL", "failure_reasons": ["validation output missing"], "checks": []}
if validation_path.exists():
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

param_report = {}
if param_json_path.exists():
    param_report = json.loads(param_json_path.read_text(encoding="utf-8"))

metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

freeze_text = freeze_txt_path.read_text(encoding="utf-8") if freeze_txt_path.exists() else ""
freeze_ok = "FREEZE_CHECK_STATUS: PASS" in freeze_text

injected_layers = []
if "Trainable parameter names:" in freeze_text:
    lines = freeze_text.splitlines()
    start = lines.index("Trainable parameter names:") + 1
    for line in lines[start:]:
        if not line.strip() or line.startswith("FREEZE_CHECK_STATUS"):
            break
        injected_layers.append(line.strip())

train_cmd = f"python scripts/train_lora_ddpm.py --config {Path(os.environ['CONFIG_PATH']).relative_to(project_root)}"
training_completed = int(os.environ["TRAIN_EXIT_CODE"]) == 0
final_status = "PASS" if training_completed and validation.get("status") == "PASS" else "FAIL"

report_lines = [
    "# Phase 2 Report — LoRA DDPM Smoke Validation",
    "",
    f"_Auto-generated at {datetime.now(timezone.utc).isoformat()}_",
    "",
    "## LoRA Injection",
    f"- Injection succeeded: {len(injected_layers) > 0}",
    f"- Target modules requested: {config.get('lora', {}).get('target_modules', [])}",
    f"- Trainable LoRA layers found: {len(injected_layers)}",
]
if injected_layers:
    report_lines.extend(["- Layers with trainable LoRA parameters:"] + [f"  - `{name}`" for name in injected_layers[:40]])

report_lines.extend(
    [
        "",
        "## Parameter Statistics",
        f"- Total parameters: {param_report.get('total_parameters', 'N/A')}",
        f"- Trainable parameters: {param_report.get('trainable_parameters', 'N/A')}",
        f"- Percent trainable: {param_report.get('percent_trainable', 'N/A')}",
        "",
        "## Freeze Verification",
        f"- Backbone freezing succeeded: {freeze_ok}",
        f"- Freeze report path: `{freeze_txt_path}`",
        "",
        "## Training",
        f"- Training command: `{train_cmd}`",
        f"- Training completed: {training_completed}",
        f"- Runtime seconds: {metrics.get('runtime_seconds', 'N/A')}",
        "",
        "## Artifacts",
        f"- Output directory: `{output_dir}`",
        f"- Sample image paths: {[str(p) for p in sample_paths] if sample_paths else 'None'}",
        f"- LoRA checkpoint path: `{adapter_dir}`",
        f"- Merged model path: `{output_dir / 'unet'}`",
        f"- Scheduler config path: `{output_dir / 'scheduler' / 'scheduler_config.json'}`",
        "",
        "## Validation",
        f"- Validation status: {validation.get('status', 'FAIL')}",
    ]
)

if validation.get("failure_reasons"):
    report_lines.append("- Warnings/Errors:")
    report_lines.extend([f"  - {reason}" for reason in validation["failure_reasons"]])
else:
    report_lines.append("- Warnings/Errors: None")

report_lines.extend(["", "## Final Status", f"**{final_status}**", ""])

report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Wrote report: {report_path}")
PY

if [[ "${VALIDATION_STATUS}" == "PASS" && "${TRAIN_EXIT_CODE}" -eq 0 ]]; then
  echo "== Final status: PASS =="
else
  echo "== Final status: FAIL =="
fi

echo "== Phase 2 LoRA smoke run complete =="
