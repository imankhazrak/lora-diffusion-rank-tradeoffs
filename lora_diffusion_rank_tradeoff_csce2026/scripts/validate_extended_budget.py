#!/usr/bin/env python3
"""Validate extended-budget DDPM artifacts."""

from __future__ import annotations

import csv
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


def has_pass_fid_rows(fid_csv: Path, required_ranks: set[int]) -> bool:
    if not fid_csv.exists():
        return False
    seen: set[int] = set()
    with fid_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rank = int(row["rank"])
            except Exception:
                continue
            if rank in required_ranks and row.get("status") == "PASS" and row.get("num_eval_images") == "2000":
                seen.add(rank)
    return seen == required_ranks


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "configs" / "extended_budget"
    logs_dir = project_root / "outputs" / "logs"
    metrics_dir = project_root / "outputs" / "metrics"
    fig_dir = project_root / "outputs" / "figures"
    validation_path = logs_dir / "extended_budget_validation.json"

    failures: list[str] = []
    checks: list[dict] = []
    required_ranks = {4, 8, 16}

    cfgs = sorted(cfg_dir.glob("rank*_20epoch.yaml"), key=lambda p: int(p.stem.split("_")[0].replace("rank", "")))
    record_check("Extended-budget config count", len(cfgs) == 3, f"count={len(cfgs)}", failures, checks)

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
        record_check(
            f"rank{rank} eval_samples_2k exists",
            (out_dir / "eval_samples_2k").is_dir(),
            str(out_dir / "eval_samples_2k"),
            failures,
            checks,
        )
        record_check(
            f"rank{rank} LoRA adapter exists",
            (out_dir / "lora_adapter" / "adapter_model.safetensors").is_file(),
            str(out_dir / "lora_adapter" / "adapter_model.safetensors"),
            failures,
            checks,
        )
        metrics_path = out_dir / "training_metrics.json"
        record_check(f"rank{rank} training_metrics exists", metrics_path.is_file(), str(metrics_path), failures, checks)
        if metrics_path.is_file():
            txt = metrics_path.read_text(encoding="utf-8", errors="ignore")
            has_nan = bool(re.search(r"\bnan\b", txt, flags=re.IGNORECASE))
            record_check(f"rank{rank} no NaN in metrics", not has_nan, "clean" if not has_nan else "nan found", failures, checks)

    ext_csv = metrics_dir / "extended_budget_results.csv"
    fid_csv = metrics_dir / "extended_budget_fid_results.csv"
    record_check("extended_budget_results.csv exists", ext_csv.is_file(), str(ext_csv), failures, checks)
    record_check("extended_budget_fid_results.csv exists", fid_csv.is_file(), str(fid_csv), failures, checks)
    record_check(
        "extended-budget FID rows complete for ranks 4/8/16",
        has_pass_fid_rows(fid_csv, required_ranks),
        str(fid_csv),
        failures,
        checks,
    )

    expected_figs = [
        "extended_budget_fid_vs_rank.png",
        "ddpm_fid_10_vs_20_epoch.png",
        "extended_budget_runtime_vs_rank.png",
        "extended_budget_training_loss_curves.png",
    ]
    for fig in expected_figs:
        record_check(f"figure {fig} exists", (fig_dir / fig).is_file(), str(fig_dir / fig), failures, checks)

    status = "PASS" if not failures else "FAIL"
    print(f"\nEXTENDED-BUDGET VALIDATION RESULT: {status}")
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
