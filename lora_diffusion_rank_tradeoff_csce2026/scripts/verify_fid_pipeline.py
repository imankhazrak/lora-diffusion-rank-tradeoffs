#!/usr/bin/env python3
"""Verify the local-folder pytorch-fid pipeline with a controlled 100-image test."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from diffusers import DDPMPipeline
from PIL import Image, UnidentifiedImageError


def load_rank_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def has_rank_artifacts(project_root: Path, rank_cfg: dict) -> bool:
    output_dir = (project_root / rank_cfg["output_dir"]).resolve()
    adapter = output_dir / "lora_adapter" / "adapter_model.safetensors"
    ckpts = sorted(output_dir.glob("checkpoint-*/lora_adapter/adapter_model.safetensors"))
    return adapter.is_file() or len(ckpts) > 0


def choose_rank_config(project_root: Path) -> tuple[int, dict]:
    cfg_dir = project_root / "configs" / "rank_sweep"
    if not torch.cuda.is_available():
        rank4_cfg = cfg_dir / "rank4.yaml"
        if rank4_cfg.exists():
            cfg = load_rank_config(rank4_cfg)
            rank4_output = (project_root / cfg["output_dir"]).resolve()
            rank4_eval = rank4_output / "eval_samples_10k"
            if rank4_eval.exists() and len(list(rank4_eval.glob("*.png"))) >= 100:
                return 4, cfg
    for rank in (2, 4):
        cfg_path = cfg_dir / f"rank{rank}.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_rank_config(cfg_path)
        if has_rank_artifacts(project_root, cfg):
            return rank, cfg
    raise RuntimeError("Neither rank2 nor rank4 has adapter/checkpoint artifacts needed for verification.")


def ensure_local_reference(project_root: Path) -> Path:
    reference_dir = project_root / "outputs" / "fid_reference" / "cifar10_test"
    pngs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
    if len(pngs) >= 10000:
        return reference_dir
    cmd = [sys.executable, str(project_root / "scripts" / "build_fid_reference.py")]
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise RuntimeError("Failed to build local CIFAR-10 reference folder.")
    pngs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
    if len(pngs) < 10000:
        raise RuntimeError(f"Reference folder incomplete: found={len(pngs)}, expected>=10000")
    return reference_dir


def generate_images(output_dir: Path, dest_dir: Path, seed: int, num_gen: int = 100, batch_size: int = 25):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for png in dest_dir.glob("*.png"):
        png.unlink()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = DDPMPipeline.from_pretrained(str(output_dir), torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    idx = 0
    while idx < num_gen:
        current_bs = min(batch_size, num_gen - idx)
        generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
        result = pipe(batch_size=current_bs, num_inference_steps=100, generator=generator)
        for image in result.images:
            image.save(dest_dir / f"sample_{idx:05d}.png")
            idx += 1


def prepare_generated_images(output_dir: Path, dest_dir: Path, seed: int, num_gen: int = 100):
    cached_eval = output_dir / "eval_samples_10k"
    cached_pngs = sorted(cached_eval.glob("*.png")) if cached_eval.exists() else []
    if not torch.cuda.is_available() and len(cached_pngs) >= num_gen:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for png in dest_dir.glob("*.png"):
            png.unlink()
        for idx, src in enumerate(cached_pngs[:num_gen]):
            shutil.copy2(src, dest_dir / f"sample_{idx:05d}.png")
        return "copied_from_eval_samples_10k"
    generate_images(output_dir, dest_dir, seed=seed, num_gen=num_gen, batch_size=25)
    return "generated_fresh"


def validate_images(image_dir: Path, expected_count: int = 100, expected_size: int = 32) -> dict:
    pngs = sorted(image_dir.glob("*.png"))
    if len(pngs) != expected_count:
        raise RuntimeError(f"Expected {expected_count} PNG files, found {len(pngs)} in {image_dir}")

    bad_files: list[str] = []
    for path in pngs:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.size != (expected_size, expected_size):
                    bad_files.append(f"{path.name}:size={img.size}")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            bad_files.append(f"{path.name}:{exc}")
    if bad_files:
        preview = "; ".join(bad_files[:5])
        raise RuntimeError(f"Corrupted/invalid images detected ({len(bad_files)}): {preview}")

    return {"count": len(pngs), "expected_size": [expected_size, expected_size], "corrupted": 0}


def write_report(report_path: Path, payload: dict):
    status = payload["final_status"]
    lines = [
        "# FID Verification Report",
        "",
        f"- Timestamp (UTC): {payload['timestamp_utc']}",
        f"- Selected rank: {payload.get('selected_rank', 'N/A')}",
        f"- Number of generated images: {payload.get('num_generated', 0)}",
        f"- Generated image directory: `{payload.get('generated_dir', 'N/A')}`",
        f"- pytorch-fid import status: {payload.get('pytorch_fid_import_status', 'FAIL')}",
        f"- Reference folder status: {payload.get('reference_status', 'UNKNOWN')}",
        f"- Reference folder: `{payload.get('reference_dir', 'N/A')}`",
        f"- FID value: {payload.get('fid_value', 'N/A')}",
        f"- Runtime seconds: {payload.get('runtime_seconds', 0)}",
        "",
        "## Warnings / Errors",
    ]
    warnings = payload.get("warnings", [])
    errors = payload.get("errors", [])
    if not warnings and not errors:
        lines.append("- None")
    else:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
        for err in errors:
            lines.append(f"- ERROR: {err}")

    lines.extend(["", "## Final Status", f"**{status}**", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.time()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs" / "fid_verification"
    json_path = out_dir / "fid_verification_result.json"
    report_path = out_dir / "fid_verification_report.md"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_rank": None,
        "num_generated": 0,
        "generated_dir": "",
        "pytorch_fid_import_status": "FAIL",
        "reference_status": "NOT_RUN",
        "reference_dir": "",
        "fid_value": None,
        "runtime_seconds": 0,
        "warnings": [],
        "errors": [],
        "final_status": "FAIL",
    }

    try:
        rank, cfg = choose_rank_config(project_root)
        payload["selected_rank"] = rank
        output_dir = (project_root / cfg["output_dir"]).resolve()
        generated_dir = out_dir / f"rank{rank}" / "generated_100"
        payload["generated_dir"] = str(generated_dir)
        seed = int(cfg.get("seed", 42))

        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths

            payload["pytorch_fid_import_status"] = "PASS"
        except Exception as exc:
            payload["errors"].append(f"pytorch-fid import failed: {exc}")
            payload["errors"].append("Install command: pip install pytorch-fid")
            raise

        try:
            reference_dir = ensure_local_reference(project_root)
            payload["reference_status"] = "READY"
            payload["reference_dir"] = str(reference_dir)
        except Exception as exc:
            payload["reference_status"] = "FAIL"
            payload["errors"].append(f"local reference folder failed: {exc}")
            raise

        payload["generation_mode"] = prepare_generated_images(output_dir, generated_dir, seed=seed, num_gen=100)
        image_check = validate_images(generated_dir, expected_count=100, expected_size=32)
        payload["num_generated"] = image_check["count"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fid_value = calculate_fid_given_paths(
            [str(reference_dir), str(generated_dir)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        fid_value = float(fid_value)
        if not math.isfinite(fid_value):
            raise RuntimeError(f"FID value is not finite: {fid_value}")
        payload["fid_value"] = fid_value
        payload["final_status"] = "PASS"
    except Exception as exc:
        if str(exc) not in "\n".join(payload["errors"]):
            payload["errors"].append(str(exc))
        payload["final_status"] = "FAIL"
    finally:
        payload["runtime_seconds"] = round(time.time() - start, 3)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_report(report_path, payload)
        print(f"Verification JSON: {json_path}")
        print(f"Verification report: {report_path}")
        print(f"Final status: {payload['final_status']}")

    if payload["final_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
