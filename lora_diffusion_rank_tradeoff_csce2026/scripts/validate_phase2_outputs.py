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
    output_dir = project_root / "outputs" / "smoke" / "lora_rank4_smoke"
    logs_dir = project_root / "outputs" / "logs"

    log_path = logs_dir / "phase2_lora_smoke.log"
    validation_json = logs_dir / "phase2_validation.json"
    param_json = logs_dir / "lora_parameter_report.json"
    param_md = logs_dir / "lora_parameter_report.md"
    freeze_txt = logs_dir / "lora_freeze_check.txt"
    metrics_json = logs_dir / "phase2_training_metrics.json"
    candidate_modules = logs_dir / "lora_candidate_modules.txt"

    sample_dir = output_dir / "generated_samples"
    adapter_dir = output_dir / "lora_adapter"
    tb_dir = output_dir / "logs"
    unet_weights = output_dir / "unet" / "diffusion_pytorch_model.safetensors"
    scheduler_cfg = output_dir / "scheduler" / "scheduler_config.json"

    failures: list[str] = []
    checks: list[dict] = []

    record_check("Output directory exists", output_dir.is_dir(), str(output_dir), failures, checks)

    record_check("Generated sample directory exists", sample_dir.is_dir(), str(sample_dir), failures, checks)
    png_samples = sorted(sample_dir.glob("*.png")) if sample_dir.is_dir() else []
    record_check("Generated sample PNG exists", len(png_samples) >= 1, f"count={len(png_samples)}", failures, checks)

    record_check("LoRA adapter directory exists", adapter_dir.is_dir(), str(adapter_dir), failures, checks)
    adapter_weight_exists = (adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()
    record_check("LoRA adapter weights exist", adapter_weight_exists, str(adapter_dir), failures, checks)
    record_check("LoRA adapter config exists", (adapter_dir / "adapter_config.json").is_file(), str(adapter_dir), failures, checks)

    checkpoint_dirs = sorted([p for p in output_dir.glob("checkpoint-*") if p.is_dir()])
    record_check(
        "Checkpoint directory exists",
        len(checkpoint_dirs) >= 1,
        f"found={[p.name for p in checkpoint_dirs]}",
        failures,
        checks,
    )

    record_check("UNet weights exist", unet_weights.is_file(), str(unet_weights), failures, checks)
    record_check("Scheduler config exists", scheduler_cfg.is_file(), str(scheduler_cfg), failures, checks)

    tb_files = sorted(tb_dir.glob("**/events.out.tfevents.*")) if tb_dir.is_dir() else []
    record_check("TensorBoard logs exist", len(tb_files) >= 1, f"count={len(tb_files)}", failures, checks)

    record_check("Parameter JSON report exists", param_json.is_file(), str(param_json), failures, checks)
    record_check("Parameter Markdown report exists", param_md.is_file(), str(param_md), failures, checks)
    record_check("Freeze check report exists", freeze_txt.is_file(), str(freeze_txt), failures, checks)
    record_check("Candidate modules report exists", candidate_modules.is_file(), str(candidate_modules), failures, checks)
    record_check("Training metrics JSON exists", metrics_json.is_file(), str(metrics_json), failures, checks)
    record_check("Phase2 smoke log exists", log_path.is_file(), str(log_path), failures, checks)

    if freeze_txt.is_file():
        text = freeze_txt.read_text(encoding="utf-8", errors="ignore")
        record_check(
            "Freeze check status PASS",
            "FREEZE_CHECK_STATUS: PASS" in text,
            "FREEZE_CHECK_STATUS token",
            failures,
            checks,
        )

    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        has_nan = bool(re.search(r"\bnan\b", log_text, flags=re.IGNORECASE))
        record_check("No NaN losses in log", not has_nan, "nan token present" if has_nan else "clean", failures, checks)

    status = "PASS" if not failures else "FAIL"
    print(f"\nVALIDATION RESULT: {status}")
    if failures:
        print("Failure reasons:")
        for item in failures:
            print(f"- {item}")

    payload = {"status": status, "checks": checks, "failure_reasons": failures}
    validation_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Validation JSON written: {validation_json}")

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
