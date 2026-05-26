#!/usr/bin/env python3
"""Compute FID for a rank run using local-folder pytorch-fid protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from diffusers import DDPMPipeline
from PIL import Image, UnidentifiedImageError

FID_HEADERS = [
    "rank",
    "fid",
    "num_eval_images",
    "protocol",
    "status",
    "error_stage",
    "error",
    "eval_dir",
    "timestamp_utc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute pytorch-fid for a rank output.")
    parser.add_argument("--config", type=str, required=True, help="Rank config YAML path")
    parser.add_argument("--num-gen", type=int, default=2000, help="Target number of generated images")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for image generation")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Delete existing eval images and regenerate from scratch.",
    )
    parser.add_argument(
        "--skip-if-existing",
        action="store_true",
        help="If a PASS row already exists for this rank and num-gen, skip work.",
    )
    parser.add_argument(
        "--fid-csv-path",
        type=str,
        default="outputs/metrics/fid_results.csv",
        help="Project-relative CSV path used to store FID rows.",
    )
    parser.add_argument(
        "--rank-log-path",
        type=str,
        default=None,
        help="Optional project-relative JSON log path for this run.",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=None,
        help="Optional project-relative directory containing generated PNGs for FID scoring.",
    )
    return parser.parse_args()


def resolve_eval_dir(output_dir: Path, num_gen: int) -> Path:
    legacy_dir = output_dir / "eval_samples_10k"
    if legacy_dir.exists() and len(list(legacy_dir.glob("*.png"))) >= num_gen:
        return legacy_dir
    if num_gen == 2000:
        target_dir = output_dir / "eval_samples_2k"
    elif num_gen == 10000:
        target_dir = output_dir / "eval_samples_10k"
    else:
        target_dir = output_dir / f"eval_samples_{num_gen}"
    return target_dir


def load_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_passing_existing_row(rows: list[dict], rank: int, num_gen: int) -> bool:
    for row in rows:
        if (
            row.get("rank") == str(rank)
            and row.get("status") == "PASS"
            and row.get("num_eval_images") == str(num_gen)
        ):
            return True
    return False


def write_fid_csv(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(csv_path)
    rows = [r for r in rows if not (r.get("rank") == str(row["rank"]) and r.get("num_eval_images") == str(row["num_eval_images"]))]
    rows.append({k: str(row.get(k, "")) for k in FID_HEADERS})
    rows.sort(key=lambda r: (int(r.get("rank", "-1")), int(r.get("num_eval_images", "0"))))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FID_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def validate_png_images(eval_dir: Path, num_gen: int, expected_size: int = 32):
    pngs = sorted(eval_dir.glob("*.png"))
    if len(pngs) < num_gen:
        raise RuntimeError(f"Not enough PNG files in {eval_dir}: found={len(pngs)}, expected>={num_gen}")

    invalid: list[str] = []
    for path in pngs[:num_gen]:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.size != (expected_size, expected_size):
                    invalid.append(f"{path.name}:size={img.size}")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            invalid.append(f"{path.name}:{exc}")

    if invalid:
        preview = "; ".join(invalid[:5])
        raise RuntimeError(f"Image validation failed for {len(invalid)} file(s): {preview}")


def ensure_local_reference(project_root: Path) -> Path:
    reference_dir = project_root / "outputs" / "fid_reference" / "cifar10_test"
    pngs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
    if len(pngs) >= 10000:
        return reference_dir
    cmd = [sys.executable, str(project_root / "scripts" / "build_fid_reference.py")]
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise RuntimeError("Failed building local CIFAR-10 reference folder")
    pngs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
    if len(pngs) < 10000:
        raise RuntimeError(f"Reference folder incomplete: {reference_dir} count={len(pngs)}")
    return reference_dir


def ensure_eval_samples(
    output_dir: Path,
    eval_dir: Path,
    num_gen: int,
    batch_size: int,
    seed: int,
    force_regenerate: bool,
):
    if force_regenerate and eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(eval_dir.glob("*.png"))
    if len(existing) >= num_gen:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Missing {num_gen - len(existing)} images in {eval_dir} and CUDA is unavailable for generation."
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = DDPMPipeline.from_pretrained(str(output_dir), torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    idx = len(existing)
    while idx < num_gen:
        current_bs = min(batch_size, num_gen - idx)
        generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
        result = pipe(batch_size=current_bs, num_inference_steps=100, generator=generator)
        for image in result.images:
            image.save(eval_dir / f"sample_{idx:05d}.png")
            idx += 1


def write_rank_log(log_path: Path, payload: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    rank = int(config.get("rank", config.get("lora", {}).get("rank", -1)))
    seed = int(config.get("seed", 42))
    output_dir = (project_root / config["output_dir"]).resolve()
    eval_dir = (project_root / args.generated_dir).resolve() if args.generated_dir else resolve_eval_dir(output_dir, args.num_gen)
    fid_csv = (project_root / args.fid_csv_path).resolve()
    if args.rank_log_path:
        rank_log = (project_root / args.rank_log_path).resolve()
    else:
        rank_log = project_root / "outputs" / "logs" / "rank_sweep" / f"rank{rank}_fid.json"

    row = {
        "rank": rank,
        "fid": "",
        "num_eval_images": args.num_gen,
        "protocol": "pytorch-fid:local-cifar10-test:dims2048",
        "status": "FAIL",
        "error_stage": "",
        "error": "",
        "eval_dir": str(eval_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    debug_payload = {
        "rank": rank,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "num_gen": args.num_gen,
        "force_regenerate": args.force_regenerate,
        "skip_if_existing": args.skip_if_existing,
        "started_at_utc": row["timestamp_utc"],
    }

    try:
        existing_rows = load_existing_rows(fid_csv)
        if args.skip_if_existing and has_passing_existing_row(existing_rows, rank, args.num_gen):
            row["status"] = "PASS"
            row["error"] = "skipped_existing_pass"
            write_rank_log(rank_log, {**debug_payload, "result": row})
            print(f"Skipping rank {rank}; PASS row already exists for num_gen={args.num_gen}.")
            return

        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths
        except Exception as exc:
            row["error_stage"] = "import"
            row["error"] = f"{exc} (install with: pip install pytorch-fid)"
            raise

        try:
            reference_dir = ensure_local_reference(project_root)
            debug_payload["reference_dir"] = str(reference_dir)
        except Exception as exc:
            row["error_stage"] = "reference_data"
            row["error"] = str(exc)
            raise

        try:
            if args.generated_dir:
                eval_dir.mkdir(parents=True, exist_ok=True)
            else:
                ensure_eval_samples(
                    output_dir=output_dir,
                    eval_dir=eval_dir,
                    num_gen=args.num_gen,
                    batch_size=args.batch_size,
                    seed=seed,
                    force_regenerate=args.force_regenerate,
                )
            validate_png_images(eval_dir, args.num_gen, expected_size=32)
        except Exception as exc:
            if not row["error_stage"]:
                row["error_stage"] = "generation_or_validation"
                row["error"] = str(exc)
            raise

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            fid_value = calculate_fid_given_paths(
                [str(reference_dir), str(eval_dir)],
                batch_size=50,
                device=device,
                dims=2048,
            )
            fid_value = float(fid_value)
            if not math.isfinite(fid_value):
                raise RuntimeError(f"FID is not finite: {fid_value}")
        except Exception as exc:
            if not row["error_stage"]:
                row["error_stage"] = "fid_compute"
                row["error"] = str(exc)
            raise

        row["fid"] = f"{fid_value:.6f}"
        row["status"] = "PASS"
        row["error_stage"] = ""
        row["error"] = ""
    except Exception:
        if not row["error_stage"]:
            row["error_stage"] = "unknown"
        write_fid_csv(row, fid_csv)
        write_rank_log(rank_log, {**debug_payload, "result": row})
        raise
    finally:
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        if row["status"] == "PASS":
            write_fid_csv(row, fid_csv)
            write_rank_log(rank_log, {**debug_payload, "result": row})

    print(f"FID rank {rank}: {row['fid']}")
    print(f"Updated: {fid_csv}")


if __name__ == "__main__":
    main()
