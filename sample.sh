#!/bin/bash
#SBATCH --job-name=smart-eval
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --account=PCS0229

set -euo pipefail

echo "=========================================="
echo "SMART Model Evaluation on GPU"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_NODELIST}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load required modules
module purge
module load cuda/12.6.2

# Activate conda environment (explicit path)
export PATH="/users/PCS0229/imankhazrak/miniconda3/bin:$PATH"
source /users/PCS0229/imankhazrak/miniconda3/etc/profile.d/conda.sh
conda activate SMART_smol-tool-agent

echo "Python: $(which python)"
python -V
echo ""

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Read args from environment (set via --export or defaults)
MODEL_PATH="${MODEL_PATH:-models/smart-qlora}"
TEST_FORMAT="${TEST_FORMAT:-jsonl}"
TEST_DATA_DIR="${TEST_DATA_DIR:-}"
COMPARE_WITH="${COMPARE_WITH:-}"
OUTPUT="${OUTPUT:-}"

echo "Model path: ${MODEL_PATH}"
echo "Test format: ${TEST_FORMAT}"
echo "Test data dir: ${TEST_DATA_DIR:-<not set, using default>}"
echo "Compare with: ${COMPARE_WITH:-<not set>}"
echo "Output file: ${OUTPUT:-<not set, will print to stdout>}"
echo ""

# GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Build command safely (no eval)
CMD=(python scripts/evaluate_model.py --model-path "${MODEL_PATH}" --test-format "${TEST_FORMAT}")

if [[ -n "${TEST_DATA_DIR}" ]]; then
  CMD+=(--test-data-dir "${TEST_DATA_DIR}")
fi

if [[ -n "${COMPARE_WITH}" ]]; then
  CMD+=(--compare-with "${COMPARE_WITH}")
fi

if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

echo "Starting evaluation command:"
printf ' %q' "${CMD[@]}"
echo -e "\n"

# Run evaluation
"${CMD[@]}"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

