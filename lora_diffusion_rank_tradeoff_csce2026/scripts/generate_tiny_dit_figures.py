#!/usr/bin/env python3
"""Generate figures for tiny DiT validation and backbone comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main():
    project_root = Path(__file__).resolve().parents[1]
    fig_dir = project_root / "outputs" / "figures"
    tiny_csv = project_root / "outputs" / "metrics" / "tiny_dit_results.csv"
    ddpm20_csv = project_root / "outputs" / "metrics" / "extended_budget_results.csv"
    cfg_dir = project_root / "configs" / "tiny_dit"

    if not tiny_csv.exists():
        raise FileNotFoundError(f"Missing tiny DiT CSV: {tiny_csv}")
    tiny_df = pd.read_csv(tiny_csv).sort_values("rank")
    tiny_df = to_numeric(tiny_df, ["rank", "fid", "runtime_seconds", "peak_gpu_memory_mb", "final_loss"])

    # 1) Tiny DiT FID vs rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tiny_df["rank"], tiny_df["fid"], marker="o", label="Tiny DiT 10 epochs")
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("FID (pytorch-fid, CIFAR10 local test)")
    ax.set_title("Tiny DiT: FID vs Rank")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, fig_dir / "tiny_dit_fid_vs_rank.png")

    # 2) Backbone comparison (DDPM 20 epochs vs Tiny DiT 10 epochs)
    if ddpm20_csv.exists():
        ddpm20_df = pd.read_csv(ddpm20_csv).sort_values("rank")
        ddpm20_df = to_numeric(ddpm20_df, ["rank", "fid"])
        ddpm20_subset = ddpm20_df[ddpm20_df["rank"].isin([4, 8, 16])].copy()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(ddpm20_subset["rank"], ddpm20_subset["fid"], marker="o", label="DDPM 20 epochs")
        ax.plot(tiny_df["rank"], tiny_df["fid"], marker="o", label="Tiny DiT 10 epochs")
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("FID")
        ax.set_title("Backbone Comparison: DDPM vs Tiny DiT")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_plot(fig, fig_dir / "ddpm_vs_tiny_dit_fid_rank4_8_16.png")

    # 3) Tiny DiT runtime vs rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tiny_df["rank"].astype(int).astype(str), tiny_df["runtime_seconds"])
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Tiny DiT Runtime vs Rank")
    save_plot(fig, fig_dir / "tiny_dit_runtime_vs_rank.png")

    # 4) Tiny DiT training loss curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg_path in sorted(cfg_dir.glob("rank*.yaml"), key=lambda p: int(p.stem.replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        metrics_path = (project_root / cfg["output_dir"] / "training_metrics.json").resolve()
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        losses = metrics.get("loss_history", [])
        if not losses:
            continue
        ax.plot(range(1, len(losses) + 1), losses, label=f"rank={rank}", linewidth=1.0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Tiny DiT Training Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, fig_dir / "tiny_dit_training_loss_curves.png")

    # 5) Tiny DiT rank sample grid
    sample_images: list[tuple[int, Image.Image]] = []
    for cfg_path in sorted(cfg_dir.glob("rank*.yaml"), key=lambda p: int(p.stem.replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        sample_dir = project_root / cfg["output_dir"] / "generated_samples"
        candidates = sorted(sample_dir.glob("sample_*.png"))
        if candidates:
            sample_images.append((rank, Image.open(candidates[0]).convert("RGB")))
    if sample_images:
        cols = len(sample_images)
        fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.5))
        if cols == 1:
            axes = [axes]
        for ax, (rank, img) in zip(axes, sample_images):
            ax.imshow(img)
            ax.set_title(f"rank={rank}", fontsize=9)
            ax.axis("off")
        save_plot(fig, fig_dir / "tiny_dit_rank_sample_comparison.png")

    print(f"Generated tiny DiT figures in: {fig_dir}")


if __name__ == "__main__":
    main()
