#!/usr/bin/env python3
"""Backfill missing Phase 3 FID scores without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Recover missing FIDs for completed Phase 3 ranks.")
    parser.add_argument("--num-gen", type=int, default=2000, help="Target eval image count for FID computation")
    parser.add_argument(
        "--skip-if-existing",
        action="store_true",
        help="Skip ranks that already have PASS rows in fid_results.csv for this num-gen",
    )
    parser.add_argument("--force-regenerate", action="store_true", help="Regenerate eval images before FID")
    parser.add_argument(
        "--allow-without-verification-pass",
        action="store_true",
        help="Bypass verification gate check (not recommended).",
    )
    return parser.parse_args()


def load_fid_rows(fid_csv: Path) -> list[dict]:
    if not fid_csv.exists():
        return []
    with fid_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_pass_row(rows: list[dict], rank: int, num_gen: int) -> bool:
    for row in rows:
        if (
            row.get("rank") == str(rank)
            and row.get("status") == "PASS"
            and row.get("num_eval_images") == str(num_gen)
        ):
            return True
    return False


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "configs" / "rank_sweep"
    fid_csv = project_root / "outputs" / "metrics" / "fid_results.csv"
    verification_json = project_root / "outputs" / "fid_verification" / "fid_verification_result.json"

    if not args.allow_without_verification_pass:
        if not verification_json.exists():
            raise SystemExit(
                f"Verification gate missing: {verification_json}. Run scripts/verify_fid_pipeline.py first."
            )
        payload = json.loads(verification_json.read_text(encoding="utf-8"))
        if payload.get("final_status") != "PASS":
            raise SystemExit(
                "Verification gate is FAIL. Run scripts/verify_fid_pipeline.py and get PASS before recovery."
            )

    recovered: list[int] = []
    skipped: list[int] = []
    missing_adapter: list[int] = []
    failed: list[int] = []

    rows = load_fid_rows(fid_csv)
    for cfg_path in sorted(cfg_dir.glob("rank*.yaml"), key=lambda p: int(p.stem.replace("rank", ""))):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        output_dir = (project_root / cfg["output_dir"]).resolve()
        adapter = output_dir / "lora_adapter" / "adapter_model.safetensors"
        if not adapter.is_file():
            missing_adapter.append(rank)
            print(f"SKIP rank{rank}: missing final adapter ({adapter})")
            continue

        if args.skip_if_existing and has_pass_row(rows, rank, args.num_gen):
            skipped.append(rank)
            print(f"SKIP rank{rank}: PASS FID already exists for num_gen={args.num_gen}")
            continue

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "compute_fid.py"),
            "--config",
            str(cfg_path),
            "--num-gen",
            str(args.num_gen),
            "--skip-if-existing",
        ]
        if args.force_regenerate:
            cmd.append("--force-regenerate")

        print(f"RUN rank{rank}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode == 0:
            recovered.append(rank)
        else:
            failed.append(rank)

    print("\nFID recovery summary")
    print(f"- recovered: {recovered if recovered else 'none'}")
    print(f"- skipped_existing: {skipped if skipped else 'none'}")
    print(f"- missing_adapter: {missing_adapter if missing_adapter else 'none'}")
    print(f"- failed: {failed if failed else 'none'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
