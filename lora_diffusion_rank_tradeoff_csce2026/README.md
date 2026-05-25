# LoRA Diffusion Rank Trade-off (CSCE 2026) - Phase 1

Phase 1 establishes a reliable baseline using the official Hugging Face Diffusers unconditional DDPM training pipeline before any LoRA rank experiments.

## Official training source

This project uses a copied official upstream script:

- `scripts/official_train_unconditional.py`

Source:

- [https://github.com/huggingface/diffusers/blob/v0.36.0/examples/unconditional_image_generation/train_unconditional.py](https://github.com/huggingface/diffusers/blob/v0.36.0/examples/unconditional_image_generation/train_unconditional.py)

No custom DDPM architecture changes are introduced in Phase 1.

## Environment setup (OSC/Linux)

```bash
conda create -n lora_diffusion_csce python=3.10 -y
conda activate lora_diffusion_csce
```

Install PyTorch (CUDA build if GPU is available on your system):

```bash
# Example (CUDA 12.1 wheels)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you need CPU-only:

```bash
pip install --upgrade pip
pip install torch torchvision
```

Install core dependencies:

```bash
pip install "diffusers[training]" accelerate datasets torchmetrics
pip install clean-fid pytorch-fid
pip install matplotlib pandas numpy tqdm
```

Configure Accelerate:

```bash
accelerate config default
```

## Run Phase 1 smoke test

From project root:

```bash
bash scripts/run_phase1_smoke.sh
```

Then validate outputs:

```bash
python scripts/validate_phase1_outputs.py
```

## Running Phase 1 on OSC

On OSC, CUDA is typically unavailable on login nodes. Run training on a GPU compute node via SLURM.

```bash
module load cuda/12.4.1
conda activate lora_diffusion_csce
accelerate config default
sbatch slurm/phase1_smoke.slurm
```

### Expected outputs

- `outputs/logs/phase1_smoke.log` (combined SLURM + training logs)
- `outputs/logs/training_metrics.json`
- `outputs/logs/phase1_validation.json`
- `outputs/smoke/ddpm_cifar10_smoke/run_metadata.json`
- `outputs/smoke/ddpm_cifar10_smoke/generated_samples/*.png`
- `outputs/phase1_report.md` (auto-generated PASS/FAIL report)

### Expected runtime

- GPU node (1 GPU): usually a few minutes for this 1-epoch smoke test.
- CPU-only fallback: can be significantly slower and may exceed practical smoke-test time.

### Troubleshooting

- If `CUDA available: False`, confirm you are running inside a GPU SLURM allocation.
- If `nvidia-smi` fails, check cluster allocation and loaded CUDA module.
- If `accelerate launch` warns about defaults, run `accelerate config default` in your environment.
- If validation fails due to missing samples/checkpoints, inspect `outputs/logs/phase1_smoke.log` first.

## Expected artifacts

- `outputs/smoke/ddpm_cifar10_smoke/` (saved model/pipeline artifacts)
- `outputs/smoke/ddpm_cifar10_smoke/generated_samples/sample_000.png`
- `outputs/logs/phase1_smoke.log`
- `outputs/phase1_report.md`
