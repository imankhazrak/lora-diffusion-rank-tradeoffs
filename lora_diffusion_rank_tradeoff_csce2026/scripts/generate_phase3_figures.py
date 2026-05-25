#!/usr/bin/env python3
"""Generate Phase 3 publication-style figures from rank sweep outputs."""

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


def main():
    project_root = Path(__file__).resolve().parents[1]
    metrics_csv = project_root / "outputs" / "metrics" / "rank_sweep_results.csv"
    fig_dir = project_root / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    df = df.sort_values("rank")
    numeric_cols = [
        "rank",
        "trainable_params",
        "runtime_seconds",
        "peak_gpu_memory_mb",
        "fid",
        "final_loss",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) FID vs Rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["rank"], df["fid"], marker="o")
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("FID (clean-fid, CIFAR10 test)")
    ax.set_title("FID vs LoRA Rank")
    ax.grid(True, alpha=0.3)
    save_plot(fig, fig_dir / "fid_vs_rank.png")

    # 2) FID vs Trainable Parameters
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["trainable_params"], df["fid"], marker="o")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("FID")
    ax.set_title("FID vs Trainable Parameters")
    ax.grid(True, alpha=0.3)
    save_plot(fig, fig_dir / "fid_vs_trainable_params.png")

    # 3) Runtime vs Rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["rank"].astype(str), df["runtime_seconds"])
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs LoRA Rank")
    save_plot(fig, fig_dir / "runtime_vs_rank.png")

    # 4) GPU Memory vs Rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["rank"].astype(str), df["peak_gpu_memory_mb"])
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("GPU Memory vs LoRA Rank")
    save_plot(fig, fig_dir / "gpu_memory_vs_rank.png")

    # 5) Training Loss Curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cfg_dir = project_root / "configs" / "rank_sweep"
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
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, fig_dir / "training_loss_curves.png")

    # 6) Rank sample comparison grid
    sample_images: list[tuple[int, Image.Image]] = []
    for cfg_path in sorted(cfg_dir.glob("rank*.yaml"), key=lambda p: int(p.stem.replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        sample_path = project_root / cfg["output_dir"] / "generated_samples" / "sample_000.png"
        if sample_path.exists():
            sample_images.append((rank, Image.open(sample_path).convert("RGB")))

    if sample_images:
        rows = 1
        cols = len(sample_images)
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5))
        if cols == 1:
            axes = [axes]
        for ax, (rank, img) in zip(axes, sample_images):
            ax.imshow(img)
            ax.set_title(f"rank={rank}", fontsize=9)
            ax.axis("off")
        save_plot(fig, fig_dir / "rank_sample_comparison.png")
    else:
        # Create placeholder so validation can fail with clear reason if needed.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No rank sample images found", ha="center", va="center")
        ax.axis("off")
        save_plot(fig, fig_dir / "rank_sample_comparison.png")

    print(f"Generated figures in: {fig_dir}")


if __name__ == "__main__":
    main()
