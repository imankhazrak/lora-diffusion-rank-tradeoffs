#!/bin/bash
# Phase 3 local-folder pytorch-fid gated recovery on OSC.
#SBATCH --job-name=phase3_localfid_recovery
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/phase3_localfid_recovery_%j.log
#SBATCH --error=outputs/logs/phase3_localfid_recovery_%j.log

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/phase3_localfid_recovery.log) 2>&1

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if type module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.4.1 || echo "WARNING: module load cuda/12.4.1 failed."
  module load miniconda3 || echo "WARNING: module load miniconda3 failed."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lora_diffusion_csce || echo "WARNING: conda activate failed; using current python."
fi

echo "== Local pytorch-fid gated recovery =="
echo "START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
python scripts/verify_fid_pipeline.py
python scripts/recover_phase3_fid.py --num-gen 2000 --skip-if-existing
python scripts/aggregate_phase3_results.py
python scripts/generate_phase3_figures.py
python scripts/validate_phase3_outputs.py
python scripts/generate_phase3_report.py
echo "END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
