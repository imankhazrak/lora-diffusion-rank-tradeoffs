#!/bin/bash
# Phase 1.1 OSC GPU smoke validation runner.
#SBATCH --job-name=phase1_ddpm_smoke
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/phase1_smoke_%j.log
#SBATCH --error=outputs/logs/phase1_smoke_%j.log

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/phase1_smoke.log) 2>&1

JOB_START_EPOCH="$(date +%s)"
echo "========== JOB START =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-UNKNOWN}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-UNKNOWN}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-UNKNOWN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Module loading can differ across OSC clusters.
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if type module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.4.1 || echo "WARNING: module load cuda/12.4.1 failed; check available module names."
  module load miniconda3 || echo "WARNING: module load miniconda3 failed; conda must already be available."
else
  echo "WARNING: module command not found; relying on existing shell environment."
fi

if ! command -v conda >/dev/null 2>&1; then
  export PATH="/users/PCS0229/imankhazrak/miniconda3/bin:${PATH}"
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lora_diffusion_csce || echo "WARNING: conda activate lora_diffusion_csce failed; continuing with current python."
else
  echo "WARNING: conda command not found. Continuing with system python."
fi

echo "========== SYSTEM INFO =========="
hostname
date
pwd

echo "========== GPU INFO =========="
nvidia-smi || true

echo "========== CUDA INFO =========="
nvcc --version || true

echo "========== PYTORCH CUDA =========="
python -c "import torch; print(torch.cuda.is_available())" || true
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" || true

echo "========== RUNTIME ENV =========="
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

bash scripts/run_phase1_smoke.sh

JOB_END_EPOCH="$(date +%s)"
ELAPSED_SECONDS=$(( JOB_END_EPOCH - JOB_START_EPOCH ))
echo "========== JOB END =========="
echo "END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "ELAPSED_SECONDS=${ELAPSED_SECONDS}"
