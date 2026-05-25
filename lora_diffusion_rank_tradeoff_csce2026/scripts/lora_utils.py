#!/usr/bin/env python3
"""Utilities for PEFT LoRA injection and verification on Diffusers UNet."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from peft import LoraConfig, get_peft_model


def discover_lora_candidate_modules(model) -> list[str]:
    """Return module names that expose common attention projection layers."""
    candidates: list[str] = []
    for name, module in model.named_modules():
        has_attn_proj = all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v", "to_out"))
        if has_attn_proj:
            candidates.append(name)
    return sorted(candidates)


def write_candidate_modules(candidates: Iterable[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [*candidates]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def inject_lora_with_peft(model, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    """Inject LoRA adapters into the provided model using PEFT."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    return get_peft_model(model, lora_config)


def count_parameters(model) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total else 0.0
    return {"total_parameters": total, "trainable_parameters": trainable, "percent_trainable": pct}


def list_trainable_layers(model) -> list[str]:
    return sorted(name for name, param in model.named_parameters() if param.requires_grad)


def freeze_backbone_verify(model) -> dict[str, int | bool]:
    named_params = list(model.named_parameters())
    trainable = [(n, p) for n, p in named_params if p.requires_grad]
    frozen = [(n, p) for n, p in named_params if not p.requires_grad]

    only_lora_trainable = True
    for name, _ in trainable:
        # PEFT names for LoRA tensors usually contain ".lora_A." or ".lora_B.".
        if ".lora_" not in name:
            only_lora_trainable = False
            break

    return {
        "frozen_param_tensors": len(frozen),
        "trainable_param_tensors": len(trainable),
        "only_lora_trainable": only_lora_trainable,
    }
