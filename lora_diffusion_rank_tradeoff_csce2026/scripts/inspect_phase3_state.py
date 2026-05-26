#!/usr/bin/env python3
"""Inspect Phase 3 rank outputs and summarize recovery state."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_fid_map(fid_csv: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    if not fid_csv.exists():
        return result
    with fid_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rank = int(row.get("rank", "-1"))
            except ValueError:
                continue
            if rank not in result or row.get("status") == "PASS":
                result[rank] = row
    return result


def detect_rank_status(project_root: Path, rank: int, cfg: dict, fid_map: dict[int, dict]) -> dict:
    output_dir = (project_root / cfg["output_dir"]).resolve()
    training_metrics = output_dir / "training_metrics.json"
    adapter_path = output_dir / "lora_adapter" / "adapter_model.safetensors"
    merged_unet = output_dir / "unet" / "diffusion_pytorch_model.safetensors"
    generated_sample = output_dir / "generated_samples" / "sample_000.png"
    checkpoints = sorted([p for p in output_dir.glob("checkpoint-*") if p.is_dir()])
    checkpoint_adapters = [
        str((ckpt / "lora_adapter" / "adapter_model.safetensors"))
        for ckpt in checkpoints
        if (ckpt / "lora_adapter" / "adapter_model.safetensors").is_file()
    ]

    eval_dir_10k = output_dir / "eval_samples_10k"
    eval_png_count = len(list(eval_dir_10k.glob("*.png"))) if eval_dir_10k.exists() else 0

    fid_row = fid_map.get(rank, {})
    fid_status = fid_row.get("status", "MISSING")
    fid_value = fid_row.get("fid", "")
    fid_error = fid_row.get("error", "")

    training_completed = False
    train_exit_code = None
    if training_metrics.exists():
        try:
            metrics_payload = json.loads(training_metrics.read_text(encoding="utf-8"))
            train_exit_code = metrics_payload.get("train_exit_code")
            training_completed = train_exit_code == 0
        except Exception:
            training_completed = False

    corruption_signals: list[str] = []
    if output_dir.exists() and not merged_unet.is_file():
        corruption_signals.append("missing_merged_unet")
    if checkpoints and not checkpoint_adapters:
        corruption_signals.append("checkpoints_without_adapter")
    if eval_dir_10k.exists() and eval_png_count == 0:
        corruption_signals.append("empty_eval_samples_10k")

    return {
        "rank": rank,
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.is_dir(),
        "training_completed": training_completed,
        "train_exit_code": train_exit_code,
        "final_adapter_exists": adapter_path.is_file(),
        "merged_unet_exists": merged_unet.is_file(),
        "generated_sample_exists": generated_sample.is_file(),
        "training_metrics_exists": training_metrics.is_file(),
        "checkpoint_count": len(checkpoints),
        "checkpoint_adapter_count": len(checkpoint_adapters),
        "eval_samples_10k_exists": eval_dir_10k.is_dir(),
        "eval_samples_10k_png_count": eval_png_count,
        "fid_status": fid_status,
        "fid_value": fid_value,
        "fid_error": fid_error,
        "corruption_signals": corruption_signals,
    }


def write_markdown(report_path: Path, rows: list[dict], timestamp: str):
    lines = [
        "# Phase 3 State Inspection",
        "",
        f"- Timestamp (UTC): {timestamp}",
        "",
        "| rank | train_done | final_adapter | metrics | sample | eval_10k_pngs | fid_status | fid | corruption_signals |",
        "|---:|:---:|:---:|:---:|:---:|---:|:---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: r["rank"]):
        lines.append(
            f"| {row['rank']} | {row['training_completed']} | {row['final_adapter_exists']} | "
            f"{row['training_metrics_exists']} | {row['generated_sample_exists']} | "
            f"{row['eval_samples_10k_png_count']} | {row['fid_status']} | {row['fid_value'] or ''} | "
            f"{', '.join(row['corruption_signals']) if row['corruption_signals'] else 'none'} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "configs" / "rank_sweep"
    logs_dir = project_root / "outputs" / "logs"
    fid_csv = project_root / "outputs" / "metrics" / "fid_results.csv"

    timestamp = datetime.now(timezone.utc).isoformat()
    fid_map = load_fid_map(fid_csv)
    rows = []
    for rank in [2, 4, 8, 16, 32]:
        cfg_path = cfg_dir / f"rank{rank}.yaml"
        if not cfg_path.exists():
            rows.append(
                {
                    "rank": rank,
                    "output_dir": "",
                    "output_dir_exists": False,
                    "training_completed": False,
                    "train_exit_code": None,
                    "final_adapter_exists": False,
                    "merged_unet_exists": False,
                    "generated_sample_exists": False,
                    "training_metrics_exists": False,
                    "checkpoint_count": 0,
                    "checkpoint_adapter_count": 0,
                    "eval_samples_10k_exists": False,
                    "eval_samples_10k_png_count": 0,
                    "fid_status": "MISSING",
                    "fid_value": "",
                    "fid_error": "missing_rank_config",
                    "corruption_signals": ["missing_rank_config"],
                }
            )
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rows.append(detect_rank_status(project_root, rank, cfg, fid_map))

    payload = {"timestamp_utc": timestamp, "ranks": rows}
    json_path = logs_dir / "phase3_state_inspection.json"
    md_path = logs_dir / "phase3_state_inspection.md"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, timestamp)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
