#!/usr/bin/env python3
"""Validate Phase 3 rank sweep outputs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml


def record_check(name: str, ok: bool, detail: str, failures: list[str], checks: list[dict]):
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name} - {detail}")
    checks.append({"name": name, "status": status, "detail": detail})
    if not ok:
        failures.append(f"{name}: {detail}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "configs" / "rank_sweep"
    logs_dir = project_root / "outputs" / "logs"
    metrics_dir = project_root / "outputs" / "metrics"
    fig_dir = project_root / "outputs" / "figures"
    validation_path = logs_dir / "phase3_validation.json"

    failures: list[str] = []
    checks: list[dict] = []

    cfgs = sorted(cfg_dir.glob("rank*.yaml"), key=lambda p: int(p.stem.replace("rank", "")))
    record_check("Rank config files present", len(cfgs) == 5, f"count={len(cfgs)}", failures, checks)

    for cfg_path in cfgs:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        rank = int(cfg["rank"])
        out_dir = (project_root / cfg["output_dir"]).resolve()

        record_check(f"rank{rank} output dir exists", out_dir.is_dir(), str(out_dir), failures, checks)
        record_check(
            f"rank{rank} sample image exists",
            (out_dir / "generated_samples" / "sample_000.png").is_file(),
            str(out_dir / "generated_samples" / "sample_000.png"),
            failures,
            checks,
        )
        ckpts = sorted([p for p in out_dir.glob("checkpoint-*") if p.is_dir()])
        record_check(f"rank{rank} checkpoint exists", len(ckpts) >= 1, f"count={len(ckpts)}", failures, checks)
        record_check(
            f"rank{rank} LoRA adapter exists",
            (out_dir / "lora_adapter" / "adapter_model.safetensors").is_file(),
            str(out_dir / "lora_adapter" / "adapter_model.safetensors"),
            failures,
            checks,
        )
        record_check(
            f"rank{rank} merged UNet exists",
            (out_dir / "unet" / "diffusion_pytorch_model.safetensors").is_file(),
            str(out_dir / "unet" / "diffusion_pytorch_model.safetensors"),
            failures,
            checks,
        )

        metrics_path = out_dir / "training_metrics.json"
        record_check(f"rank{rank} training_metrics exists", metrics_path.is_file(), str(metrics_path), failures, checks)
        if metrics_path.is_file():
            txt = metrics_path.read_text(encoding="utf-8", errors="ignore")
            has_nan = bool(re.search(r"\bnan\b", txt, flags=re.IGNORECASE))
            record_check(f"rank{rank} no NaN in metrics", not has_nan, "clean" if not has_nan else "nan found", failures, checks)

    fid_csv = metrics_dir / "fid_results.csv"
    sweep_csv = metrics_dir / "rank_sweep_results.csv"
    record_check("fid_results.csv exists", fid_csv.is_file(), str(fid_csv), failures, checks)
    record_check("rank_sweep_results.csv exists", sweep_csv.is_file(), str(sweep_csv), failures, checks)

    expected_figs = [
        "fid_vs_rank.png",
        "fid_vs_trainable_params.png",
        "runtime_vs_rank.png",
        "gpu_memory_vs_rank.png",
        "training_loss_curves.png",
        "rank_sample_comparison.png",
    ]
    for fig in expected_figs:
        record_check(f"figure {fig} exists", (fig_dir / fig).is_file(), str(fig_dir / fig), failures, checks)

    status = "PASS" if not failures else "FAIL"
    print(f"\nVALIDATION RESULT: {status}")
    if failures:
        print("Failure reasons:")
        for f in failures:
            print(f"- {f}")

    payload = {"status": status, "checks": checks, "failure_reasons": failures}
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Validation JSON written: {validation_path}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
