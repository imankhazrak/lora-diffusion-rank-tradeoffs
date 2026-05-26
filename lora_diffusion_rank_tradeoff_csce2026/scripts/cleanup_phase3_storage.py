#!/usr/bin/env python3
"""Safe cleanup for Phase 3 storage pressure."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Cleanup reproducible intermediate Phase 3 artifacts.")
    parser.add_argument("--confirm", action="store_true", help="Actually delete files/directories.")
    return parser.parse_args()


def dir_size_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def select_paths_for_cleanup(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    phase3_dir = project_root / "outputs" / "phase3"
    if not phase3_dir.exists():
        return candidates

    for rank_dir in sorted([p for p in phase3_dir.glob("rank*") if p.is_dir()]):
        # Keep top-level lora_adapter, generated_samples, and metrics/report files.
        checkpoints = sorted([p for p in rank_dir.glob("checkpoint-*") if p.is_dir()])
        if len(checkpoints) > 1:
            for ckpt in checkpoints[:-1]:
                candidates.append(ckpt)

        # Temporary FID image caches can be regenerated.
        for eval_dir in rank_dir.glob("eval_samples_*"):
            if eval_dir.is_dir():
                candidates.append(eval_dir)

    # Duplicate per-rank logs are safe to remove if needed.
    rank_log_dir = project_root / "outputs" / "logs" / "rank_sweep"
    if rank_log_dir.exists():
        for log_json in rank_log_dir.glob("rank*_fid.json"):
            candidates.append(log_json)

    return candidates


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    targets = select_paths_for_cleanup(project_root)
    unique_targets: list[Path] = []
    seen = set()
    for path in targets:
        if path.exists() and str(path) not in seen:
            unique_targets.append(path)
            seen.add(str(path))

    if not unique_targets:
        print("No cleanup candidates found.")
        return

    planned_bytes = sum(dir_size_bytes(path) for path in unique_targets)
    print("Cleanup candidates:")
    for path in unique_targets:
        print(f"- {path} ({format_size(dir_size_bytes(path))})")
    print(f"Estimated reclaimable: {format_size(planned_bytes)}")

    if not args.confirm:
        print("Dry run only. Re-run with --confirm to delete.")
        return

    reclaimed = 0
    for path in unique_targets:
        bytes_here = dir_size_bytes(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        reclaimed += bytes_here
        print(f"Deleted: {path}")

    print(f"Reclaimed: {format_size(reclaimed)}")


if __name__ == "__main__":
    main()
