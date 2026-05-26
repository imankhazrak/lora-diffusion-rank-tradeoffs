#!/usr/bin/env python3
"""Generate figures for extended-budget DDPM validation (ranks 4,8,16)."""

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
    ext_csv = project_root / "outputs" / "metrics" / "extended_budget_results.csv"
    main_csv = project_root / "outputs" / "metrics" / "rank_sweep_results.csv"
    cfg_dir = project_root / "configs" / "extended_budget"

    if not ext_csv.exists():
        raise FileNotFoundError(f"Missing extended-budget CSV: {ext_csv}")
    if not main_csv.exists():
        raise FileNotFoundError(f"Missing main DDPM CSV: {main_csv}")

    ext_df = pd.read_csv(ext_csv).sort_values("rank")
    ext_df = to_numeric(ext_df, ["rank", "fid", "runtime_seconds", "peak_gpu_memory_mb", "final_loss"])
    main_df = pd.read_csv(main_csv).sort_values("rank")
    main_df = to_numeric(main_df, ["rank", "fid"])
    main_subset = main_df[main_df["rank"].isin([4, 8, 16])].copy()

    # 1) Extended-budget FID vs rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ext_df["rank"], ext_df["fid"], marker="o", label="DDPM 20 epochs")
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("FID (pytorch-fid, CIFAR10 local test)")
    ax.set_title("Extended-Budget DDPM: FID vs Rank (20 epochs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, fig_dir / "extended_budget_fid_vs_rank.png")

    # 2) DDPM 10 vs 20 epochs (rank 4/8/16)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(main_subset["rank"], main_subset["fid"], marker="o", label="DDPM 10 epochs")
    ax.plot(ext_df["rank"], ext_df["fid"], marker="o", label="DDPM 20 epochs")
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("FID")
    ax.set_title("DDPM FID Comparison: 10 vs 20 epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, fig_dir / "ddpm_fid_10_vs_20_epoch.png")

    # 3) Extended-budget runtime vs rank
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ext_df["rank"].astype(int).astype(str), ext_df["runtime_seconds"])
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Extended-Budget DDPM Runtime vs Rank")
    save_plot(fig, fig_dir / "extended_budget_runtime_vs_rank.png")

    # 4) Extended-budget training loss curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg_path in sorted(cfg_dir.glob("rank*_20epoch.yaml"), key=lambda p: int(p.stem.split("_")[0].replace("rank", ""))):
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
    ax.set_title("Extended-Budget DDPM Training Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, fig_dir / "extended_budget_training_loss_curves.png")

    # 5) Extended-budget rank sample grid
    sample_images: list[tuple[int, Image.Image]] = []
    for cfg_path in sorted(cfg_dir.glob("rank*_20epoch.yaml"), key=lambda p: int(p.stem.split("_")[0].replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        sample_path = project_root / cfg["output_dir"] / "generated_samples" / "sample_000.png"
        if sample_path.exists():
            sample_images.append((rank, Image.open(sample_path).convert("RGB")))
    if sample_images:
        cols = len(sample_images)
        fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.5))
        if cols == 1:
            axes = [axes]
        for ax, (rank, img) in zip(axes, sample_images):
            ax.imshow(img)
            ax.set_title(f"rank={rank}", fontsize=9)
            ax.axis("off")
        save_plot(fig, fig_dir / "extended_budget_rank_sample_comparison.png")

    print(f"Generated extended-budget figures in: {fig_dir}")


if __name__ == "__main__":
    main()
