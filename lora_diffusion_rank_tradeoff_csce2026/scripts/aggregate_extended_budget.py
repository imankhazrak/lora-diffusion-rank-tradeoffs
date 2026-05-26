#!/usr/bin/env python3
"""Aggregate extended-budget DDPM metrics into CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def load_fid_map(fid_csv: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    if not fid_csv.exists():
        return values
    with fid_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("status") == "PASS" and row.get("num_eval_images") == "2000":
                    values[int(row["rank"])] = float(row["fid"])
            except Exception:
                continue
    return values


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "configs" / "extended_budget"
    out_csv = project_root / "outputs" / "metrics" / "extended_budget_results.csv"
    fid_csv = project_root / "outputs" / "metrics" / "extended_budget_fid_results.csv"
    fid_map = load_fid_map(fid_csv)

    rows: list[dict] = []
    for cfg_path in sorted(cfg_dir.glob("rank*_20epoch.yaml"), key=lambda p: int(p.stem.split("_")[0].replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        epochs = int(cfg.get("epochs", 20))
        out_dir = (project_root / cfg["output_dir"]).resolve()

        param_path = out_dir / "lora_parameter_report.json"
        metrics_path = out_dir / "training_metrics.json"
        total_params = ""
        trainable_params = ""
        pct_trainable = ""
        runtime_seconds = ""
        avg_epoch_time = ""
        peak_gpu_memory_mb = ""
        final_loss = ""

        if param_path.exists():
            p = json.loads(param_path.read_text(encoding="utf-8"))
            total_params = p.get("total_parameters", "")
            trainable_params = p.get("trainable_parameters", "")
            pct_trainable = p.get("percent_trainable", "")

        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            runtime_seconds = m.get("runtime_seconds", "")
            avg_epoch_time = round(float(runtime_seconds) / max(1, epochs), 4) if runtime_seconds != "" else ""
            peak_gpu_memory_mb = m.get("max_gpu_memory_mb_observed", "")
            final_loss = m.get("loss_last", "")

        rows.append(
            {
                "rank": rank,
                "epochs": epochs,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "percent_trainable": pct_trainable,
                "runtime_seconds": runtime_seconds,
                "avg_epoch_time": avg_epoch_time,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "final_loss": final_loss,
                "fid": fid_map.get(rank, ""),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "epochs",
                "total_params",
                "trainable_params",
                "percent_trainable",
                "runtime_seconds",
                "avg_epoch_time",
                "peak_gpu_memory_mb",
                "final_loss",
                "fid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregate metrics: {out_csv}")


if __name__ == "__main__":
    main()
