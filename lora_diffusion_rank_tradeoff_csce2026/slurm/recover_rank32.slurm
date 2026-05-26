#!/bin/bash
# Prepare rank32 retraining recovery on OSC.
#SBATCH --job-name=recover_rank32
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/recover_rank32_%j.log
#SBATCH --error=outputs/logs/recover_rank32_%j.log

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p outputs/logs
exec > >(tee -a outputs/logs/recover_rank32.log) 2>&1

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

echo "recover_rank32.slurm is ready."
echo "Manual submission required by policy."
echo "Would run: bash scripts/recover_rank32.sh"

bash scripts/recover_rank32.sh
