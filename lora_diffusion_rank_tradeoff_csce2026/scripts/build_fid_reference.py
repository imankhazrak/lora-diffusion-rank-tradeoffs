#!/usr/bin/env python3
"""Build a local CIFAR-10 test PNG reference folder for pytorch-fid."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from PIL import Image, UnidentifiedImageError


def parse_args():
    parser = argparse.ArgumentParser(description="Build local CIFAR-10 test reference images.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing reference PNGs and rebuild from scratch.",
    )
    return parser.parse_args()


def validate_reference_images(reference_dir: Path) -> tuple[int, int]:
    pngs = sorted(reference_dir.glob("*.png"))
    bad = 0
    for path in pngs:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.mode != "RGB" or img.size != (32, 32):
                    bad += 1
        except (OSError, ValueError, UnidentifiedImageError):
            bad += 1
    return len(pngs), bad


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    reference_root = project_root / "outputs" / "fid_reference"
    reference_dir = reference_root / "cifar10_test"
    metadata_path = reference_root / "cifar10_test_metadata.json"

    if args.force_rebuild and reference_dir.exists():
        for path in reference_dir.glob("*.png"):
            path.unlink()
    reference_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("cifar10", split="test")
    expected_count = len(dataset)

    existing_count, existing_bad = validate_reference_images(reference_dir)
    rebuilt = False
    if existing_count != expected_count or existing_bad > 0:
        for path in reference_dir.glob("*.png"):
            path.unlink()
        for idx, img in enumerate(dataset["img"]):
            out_path = reference_dir / f"test_{idx:05d}.png"
            img.convert("RGB").save(out_path)
        rebuilt = True

    final_count, final_bad = validate_reference_images(reference_dir)
    if final_count != expected_count:
        raise RuntimeError(f"Reference count mismatch: expected={expected_count}, found={final_count}")
    if final_bad > 0:
        raise RuntimeError(f"Reference validation failed: corrupted_or_bad={final_bad}")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "cifar10",
        "split": "test",
        "reference_dir": str(reference_dir),
        "expected_count": expected_count,
        "actual_count": final_count,
        "validated_pngs": final_count,
        "bad_images": final_bad,
        "image_mode": "RGB",
        "image_size": [32, 32],
        "rebuilt": rebuilt,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Reference ready: {reference_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
