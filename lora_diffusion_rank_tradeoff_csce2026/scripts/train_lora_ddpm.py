#!/usr/bin/env python3
"""Phase 2 LoRA smoke training on top of Phase 1 DDPM baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import accelerate
import datasets
import diffusers
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

from lora_utils import (
    count_parameters,
    discover_lora_candidate_modules,
    freeze_backbone_verify,
    inject_lora_with_peft,
    list_trainable_layers,
    write_candidate_modules,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on DDPM UNet (smoke run).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (project_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_dir = (project_root / config.get("base_model_dir", "outputs/smoke/ddpm_cifar10_smoke")).resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_name = config.get("dataset", "cifar10")
    resolution = int(config.get("resolution", 32))
    batch_size = int(config.get("batch_size", 32))
    num_epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 1e-4))

    lora_cfg = config.get("lora", {})
    rank = int(lora_cfg.get("rank", 4))
    alpha = int(lora_cfg.get("alpha", 4))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])

    mixed_precision = config.get("mixed_precision", "auto")
    if mixed_precision == "auto":
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    amp_enabled = mixed_precision == "fp16" and torch.cuda.is_available()

    print("== Phase 2 LoRA training setup ==")
    print(f"Config path: {config_path}")
    print(f"Base model dir: {base_model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"LoRA targets: {target_modules}")

    unet = UNet2DModel.from_pretrained(str(base_model_dir), subfolder="unet")

    # Record candidate attention modules before LoRA injection.
    candidates = discover_lora_candidate_modules(unet)
    candidate_modules_path = logs_dir / "lora_candidate_modules.txt"
    write_candidate_modules(candidates, candidate_modules_path)
    write_candidate_modules(candidates, output_dir / "lora_candidate_modules.txt")
    print(f"Wrote LoRA candidate modules: {candidate_modules_path}")

    model = inject_lora_with_peft(
        model=unet,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    param_stats = count_parameters(model)
    trainable_names = list_trainable_layers(model)
    freeze_stats = freeze_backbone_verify(model)

    lora_param_json = logs_dir / "lora_parameter_report.json"
    lora_param_md = logs_dir / "lora_parameter_report.md"
    lora_freeze_txt = logs_dir / "lora_freeze_check.txt"
    lora_param_json_local = output_dir / "lora_parameter_report.json"
    lora_param_md_local = output_dir / "lora_parameter_report.md"
    lora_freeze_txt_local = output_dir / "lora_freeze_check.txt"

    param_payload = {
        "model": "DDPM_UNet2D_with_LoRA",
        "total_parameters": param_stats["total_parameters"],
        "trainable_parameters": param_stats["trainable_parameters"],
        "percent_trainable": round(param_stats["percent_trainable"], 6),
        "target_modules": target_modules,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    lora_param_json.write_text(json.dumps(param_payload, indent=2), encoding="utf-8")
    lora_param_json_local.write_text(json.dumps(param_payload, indent=2), encoding="utf-8")

    md_lines = [
        "# LoRA Parameter Report",
        "",
        "| Model | Total Params | Trainable Params | Percent Trainable |",
        "|---|---:|---:|---:|",
        (
            f"| DDPM_UNet2D_with_LoRA | {param_stats['total_parameters']} | "
            f"{param_stats['trainable_parameters']} | {param_stats['percent_trainable']:.6f}% |"
        ),
        "",
    ]
    lora_param_md.write_text("\n".join(md_lines), encoding="utf-8")
    lora_param_md_local.write_text("\n".join(md_lines), encoding="utf-8")

    freeze_lines = [
        "LoRA Freeze Verification",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"frozen_param_tensors: {freeze_stats['frozen_param_tensors']}",
        f"trainable_param_tensors: {freeze_stats['trainable_param_tensors']}",
        f"only_lora_trainable: {freeze_stats['only_lora_trainable']}",
        "",
        "Trainable parameter names:",
        *trainable_names,
        "",
        "FREEZE_CHECK_STATUS: PASS" if freeze_stats["only_lora_trainable"] else "FREEZE_CHECK_STATUS: FAIL",
    ]
    lora_freeze_txt.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")
    lora_freeze_txt_local.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    noise_scheduler = DDPMScheduler.from_pretrained(str(base_model_dir), subfolder="scheduler")

    augment = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = load_dataset(dataset_name, split="train")
    image_column = "image" if "image" in dataset.column_names else "img"

    def transform_images(examples):
        imgs = [augment(img.convert("RGB")) for img in examples[image_column]]
        return {"input": imgs}

    dataset.set_transform(transform_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    tb_dir = output_dir / "logs" / "train_lora_ddpm"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    losses: list[float] = []
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            clean_images = batch["input"].to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                model_out = model(noisy_images, timesteps).sample
                loss = F.mse_loss(model_out.float(), noise.float())

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1
            loss_val = float(loss.detach().item())
            losses.append(loss_val)
            writer.add_scalar("train/loss", loss_val, global_step)
            pbar.set_postfix(loss=loss_val, step=global_step)

    writer.flush()
    writer.close()

    runtime_seconds = max(1, int(time.time() - start_time))
    steps_completed = global_step
    samples_processed = steps_completed * batch_size

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter: {adapter_dir}")

    checkpoint_dir = output_dir / f"checkpoint-{max(1, global_step)}"
    checkpoint_adapter_dir = checkpoint_dir / "lora_adapter"
    checkpoint_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_adapter_dir))
    print(f"Saved LoRA checkpoint adapter: {checkpoint_adapter_dir}")

    # Merge adapter into base weights for sample generation and pipeline export.
    merged_unet = model.merge_and_unload()
    merged_unet = merged_unet.to(device)
    merged_unet.eval()

    pipeline = DDPMPipeline(unet=merged_unet, scheduler=noise_scheduler)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(str(output_dir))

    sample_dir = output_dir / "generated_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    sample = pipeline(batch_size=1, num_inference_steps=100, generator=generator).images[0]
    sample_path = sample_dir / "sample_000.png"
    sample.save(sample_path)
    print(f"Saved sample image: {sample_path}")

    train_metrics = {
        "train_exit_code": 0,
        "runtime_seconds": runtime_seconds,
        "steps_completed": steps_completed,
        "samples_processed_estimate": samples_processed,
        "samples_per_second_estimate": round(samples_processed / runtime_seconds, 4),
        "loss_history": losses,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_last": losses[-1] if losses else None,
        "max_gpu_memory_mb_observed": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        if torch.cuda.is_available()
        else 0.0,
    }
    metrics_path = logs_dir / "phase2_training_metrics.json"
    metrics_path.write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    (output_dir / "training_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "accelerate_version": accelerate.__version__,
        "datasets_version": datasets.__version__,
        "dataset": dataset_name,
        "hyperparameters": {
            "resolution": resolution,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "mixed_precision": mixed_precision,
            "seed": seed,
            "target_modules": target_modules,
        },
        "base_model_dir": str(base_model_dir),
        "lora_adapter_dir": str(adapter_dir),
        "sample_image": str(sample_path),
    }
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Wrote run metadata: {run_metadata_path}")


if __name__ == "__main__":
    main()
