#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def record_check(name: str, ok: bool, detail: str, failures: list[str], checks: list[dict]):
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name} - {detail}")
    checks.append({"name": name, "status": status, "detail": detail})
    if not ok:
        failures.append(f"{name}: {detail}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    smoke_dir = project_root / "outputs" / "smoke" / "ddpm_cifar10_smoke"
    log_path = project_root / "outputs" / "logs" / "phase1_smoke.log"
    validation_json = project_root / "outputs" / "logs" / "phase1_validation.json"
    tb_dir = smoke_dir / "logs"
    sample_dir = smoke_dir / "generated_samples"

    failures: list[str] = []
    checks: list[dict] = []

    record_check("Output directory exists", smoke_dir.is_dir(), str(smoke_dir), failures, checks)
    record_check("Generated samples directory exists", sample_dir.is_dir(), str(sample_dir), failures, checks)

    png_samples = sorted(sample_dir.glob("*.png")) if sample_dir.is_dir() else []
    record_check(
        "At least one PNG sample exists",
        len(png_samples) >= 1,
        f"count={len(png_samples)}",
        failures,
        checks,
    )

    checkpoint_dirs = sorted(smoke_dir.glob("checkpoint-*")) if smoke_dir.is_dir() else []
    checkpoint_dirs = [p for p in checkpoint_dirs if p.is_dir()]
    record_check(
        "Checkpoint directory exists",
        len(checkpoint_dirs) >= 1,
        f"found={[p.name for p in checkpoint_dirs[:3]]}",
        failures,
        checks,
    )

    unet_weights = smoke_dir / "unet" / "diffusion_pytorch_model.safetensors"
    scheduler_cfg = smoke_dir / "scheduler" / "scheduler_config.json"
    record_check("UNet weights exist", unet_weights.is_file(), str(unet_weights), failures, checks)
    record_check("Scheduler config exists", scheduler_cfg.is_file(), str(scheduler_cfg), failures, checks)

    tb_files = sorted(tb_dir.glob("**/events.out.tfevents.*")) if tb_dir.is_dir() else []
    record_check(
        "TensorBoard logs exist",
        len(tb_files) >= 1,
        f"count={len(tb_files)} in {tb_dir}",
        failures,
        checks,
    )

    record_check("Smoke log exists", log_path.is_file(), str(log_path), failures, checks)
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        has_nan = bool(re.search(r"\bnan\b", text, flags=re.IGNORECASE))
        record_check("No NaN in log", not has_nan, "nan token present" if has_nan else "clean", failures, checks)

    validation_status = "PASS" if not failures else "FAIL"
    print(f"\nVALIDATION RESULT: {validation_status}")
    if failures:
        print("Failure reasons:")
        for item in failures:
            print(f"- {item}")

    payload = {"status": validation_status, "checks": checks, "failure_reasons": failures}
    validation_json.parent.mkdir(parents=True, exist_ok=True)
    validation_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Validation JSON written: {validation_json}")

    sys.exit(0 if validation_status == "PASS" else 1)


if __name__ == "__main__":
    main()
