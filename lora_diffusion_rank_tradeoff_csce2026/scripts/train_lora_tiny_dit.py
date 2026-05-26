#!/usr/bin/env python3
"""Train LoRA adapters on a tiny DiT-style Transformer2DModel backbone."""

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
from diffusers import DDPMScheduler, Transformer2DModel
from PIL import Image
from torch.utils.data import DataLoader
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
    parser = argparse.ArgumentParser(description="Train Tiny DiT LoRA validation run.")
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


def save_images(images_tensor: torch.Tensor, output_dir: Path, start_idx: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    image_batch = (images_tensor / 2 + 0.5).clamp(0, 1)
    image_batch = (image_batch * 255).round().to(torch.uint8).cpu()
    for i in range(image_batch.shape[0]):
        arr = image_batch[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(output_dir / f"sample_{start_idx + i:05d}.png")
        count += 1
    return count


def generate_samples(
    model,
    scheduler: DDPMScheduler,
    output_dir: Path,
    seed: int,
    total_images: int,
    batch_size: int,
    resolution: int,
    in_channels: int = 3,
    use_timestep_cond: bool = False,
):
    device = next(model.parameters()).device
    model.eval()
    existing = sorted(output_dir.glob("*.png"))
    start = len(existing)
    idx = start
    with torch.no_grad():
        pbar = tqdm(total=total_images - start, desc=f"Generate {output_dir.name}", leave=False)
        while idx < total_images:
            current_bs = min(batch_size, total_images - idx)
            gen = torch.Generator(device=device).manual_seed(seed + idx)
            latents = torch.randn((current_bs, in_channels, resolution, resolution), generator=gen, device=device)
            scheduler.set_timesteps(100, device=device)
            for t in scheduler.timesteps:
                if use_timestep_cond:
                    timestep = torch.full((current_bs,), int(t.item()) if hasattr(t, "item") else int(t), device=device, dtype=torch.long)
                    model_out = model(latents, timestep=timestep).sample
                else:
                    model_out = model(latents).sample
                latents = scheduler.step(model_out, t, latents).prev_sample
            wrote = save_images(latents, output_dir, idx)
            idx += wrote
            pbar.update(wrote)
        pbar.close()
    model.train()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (project_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_name = config.get("dataset", "cifar10")
    resolution = int(config.get("resolution", 32))
    batch_size = int(config.get("batch_size", 64))
    num_epochs = int(config.get("epochs", 10))
    learning_rate = float(config.get("learning_rate", 1e-4))
    eval_num_gen = int(config.get("eval_num_gen", 2000))
    eval_batch_size = int(config.get("eval_batch_size", 64))

    lora_cfg = config.get("lora", {})
    rank = int(lora_cfg.get("rank", 4))
    alpha = int(lora_cfg.get("alpha", rank))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])

    tiny_cfg = config.get("tiny_dit", {})
    in_channels = int(tiny_cfg.get("in_channels", 3))
    out_channels = int(tiny_cfg.get("out_channels", 3))
    num_layers = int(tiny_cfg.get("num_layers", 4))
    num_heads = int(tiny_cfg.get("num_attention_heads", 4))
    head_dim = int(tiny_cfg.get("attention_head_dim", 32))
    num_embeds_ada_norm = int(tiny_cfg.get("num_embeds_ada_norm", 1000))
    norm_num_groups = int(tiny_cfg.get("norm_num_groups", 1))
    if in_channels % max(1, norm_num_groups) != 0:
        # GroupNorm requires channels divisible by groups; fall back safely.
        norm_num_groups = 1
    norm_type = str(tiny_cfg.get("norm_type", "layer_norm"))
    use_timestep_cond = norm_type.startswith("ada_norm")

    mixed_precision = config.get("mixed_precision", "auto")
    if mixed_precision == "auto":
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    amp_enabled = mixed_precision == "fp16" and torch.cuda.is_available()

    print("== Tiny DiT LoRA training setup ==")
    print(f"Config path: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"LoRA targets: {target_modules}")

    model_kwargs = {
        "sample_size": resolution,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "norm_num_groups": norm_num_groups,
        "norm_type": norm_type,
    }
    if use_timestep_cond:
        model_kwargs["num_embeds_ada_norm"] = num_embeds_ada_norm
    backbone = Transformer2DModel(**model_kwargs)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    candidates = discover_lora_candidate_modules(backbone)
    candidate_modules_path = output_dir / "lora_candidate_modules.txt"
    write_candidate_modules(candidates, candidate_modules_path)

    model = inject_lora_with_peft(
        model=backbone,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    param_stats = count_parameters(model)
    trainable_names = list_trainable_layers(model)
    freeze_stats = freeze_backbone_verify(model)

    param_payload = {
        "model": "TinyDiT_Transformer2D_with_LoRA",
        "total_parameters": param_stats["total_parameters"],
        "trainable_parameters": param_stats["trainable_parameters"],
        "percent_trainable": round(param_stats["percent_trainable"], 6),
        "target_modules": target_modules,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "lora_parameter_report.json").write_text(json.dumps(param_payload, indent=2), encoding="utf-8")
    (output_dir / "lora_parameter_report.md").write_text(
        "\n".join(
            [
                "# LoRA Parameter Report",
                "",
                "| Model | Total Params | Trainable Params | Percent Trainable |",
                "|---|---:|---:|---:|",
                (
                    f"| TinyDiT_Transformer2D_with_LoRA | {param_stats['total_parameters']} | "
                    f"{param_stats['trainable_parameters']} | {param_stats['percent_trainable']:.6f}% |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "lora_freeze_check.txt").write_text(
        "\n".join(
            [
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
        )
        + "\n",
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

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

    losses: list[float] = []
    global_step = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            clean_images = batch["input"].to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if use_timestep_cond:
                    model_out = model(noisy_images, timestep=timesteps).sample
                else:
                    model_out = model(noisy_images).sample
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
            pbar.set_postfix(loss=loss_val, step=global_step)

    runtime_seconds = max(1, int(time.time() - start_time))
    steps_completed = global_step
    samples_processed = steps_completed * batch_size

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))

    checkpoint_dir = output_dir / f"checkpoint-{max(1, global_step)}" / "lora_adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))

    merged_model = model.merge_and_unload()
    merged_model.eval().to(device)
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(transformer_dir))
    noise_scheduler.save_pretrained(str(output_dir / "scheduler"))

    sample_dir = output_dir / "generated_samples"
    generate_samples(
        model=merged_model,
        scheduler=noise_scheduler,
        output_dir=sample_dir,
        seed=seed,
        total_images=1,
        batch_size=1,
        resolution=resolution,
        in_channels=in_channels,
        use_timestep_cond=use_timestep_cond,
    )
    eval_dir = output_dir / "eval_samples_2k"
    generate_samples(
        model=merged_model,
        scheduler=noise_scheduler,
        output_dir=eval_dir,
        seed=seed + 1000,
        total_images=eval_num_gen,
        batch_size=eval_batch_size,
        resolution=resolution,
        in_channels=in_channels,
        use_timestep_cond=use_timestep_cond,
    )

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
            "tiny_dit": tiny_cfg,
            "use_timestep_cond": use_timestep_cond,
        },
        "lora_adapter_dir": str(adapter_dir),
        "sample_image": str(sample_dir / "sample_00000.png"),
        "eval_dir": str(eval_dir),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Completed Tiny DiT run: {output_dir}")


if __name__ == "__main__":
    main()
