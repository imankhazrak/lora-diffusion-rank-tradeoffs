#!/usr/bin/env python3
"""Compute clean-fid for a rank run using CIFAR-10 test protocol."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from diffusers import DDPMPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Compute clean-fid for a rank output.")
    parser.add_argument("--config", type=str, required=True, help="Rank config YAML path")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of generated images for FID")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for image generation")
    return parser.parse_args()


def ensure_eval_samples(output_dir: Path, num_samples: int, batch_size: int, seed: int):
    eval_dir = output_dir / "eval_samples_10k"
    eval_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(eval_dir.glob("*.png"))
    if len(existing) >= num_samples:
        return eval_dir

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = DDPMPipeline.from_pretrained(str(output_dir), torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    start_idx = len(existing)
    idx = start_idx
    while idx < num_samples:
        current_bs = min(batch_size, num_samples - idx)
        generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
        result = pipe(batch_size=current_bs, num_inference_steps=100, generator=generator)
        for image in result.images:
            image.save(eval_dir / f"sample_{idx:05d}.png")
            idx += 1
    return eval_dir


def write_fid_csv(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["rank", "fid", "num_eval_images", "protocol", "status", "error", "timestamp_utc"]
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("rank") != str(row["rank"])]
    rows.append({k: str(row.get(k, "")) for k in headers})
    rows.sort(key=lambda r: int(r["rank"]))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    rank = int(config.get("rank", config.get("lora", {}).get("rank", -1)))
    seed = int(config.get("seed", 42))
    output_dir = (project_root / config["output_dir"]).resolve()
    fid_csv = project_root / "outputs" / "metrics" / "fid_results.csv"

    row = {
        "rank": rank,
        "fid": "",
        "num_eval_images": args.num_samples,
        "protocol": "clean-fid:cifar10:test:10k",
        "status": "FAIL",
        "error": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from cleanfid import fid

        eval_dir = ensure_eval_samples(output_dir, args.num_samples, args.batch_size, seed)
        fid_value = fid.compute_fid(
            str(eval_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="test",
            mode="clean",
        )
        row["fid"] = f"{float(fid_value):.6f}"
        row["status"] = "PASS"
    except Exception as exc:  # pragma: no cover
        row["error"] = str(exc)
        write_fid_csv(row, fid_csv)
        raise

    write_fid_csv(row, fid_csv)
    print(f"FID rank {rank}: {row['fid']}")
    print(f"Updated: {fid_csv}")


if __name__ == "__main__":
    main()
