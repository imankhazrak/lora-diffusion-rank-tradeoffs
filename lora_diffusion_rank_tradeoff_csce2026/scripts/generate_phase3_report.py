#!/usr/bin/env python3
"""Generate phase3_report.md from metrics, validation, and logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    metrics_csv = project_root / "outputs" / "metrics" / "rank_sweep_results.csv"
    fid_csv = project_root / "outputs" / "metrics" / "fid_results.csv"
    validation_json = project_root / "outputs" / "logs" / "phase3_validation.json"
    report_path = project_root / "outputs" / "phase3_report.md"

    validation = {"status": "FAIL", "failure_reasons": ["validation file missing"]}
    if validation_json.exists():
        validation = json.loads(validation_json.read_text(encoding="utf-8"))

    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
    else:
        df = pd.DataFrame()

    if fid_csv.exists():
        fid_df = pd.read_csv(fid_csv)
    else:
        fid_df = pd.DataFrame()

    completed_ranks = []
    failed_ranks = []
    unstable_ranks = []
    best_rank = "N/A"

    if not df.empty:
        for _, row in df.iterrows():
            rank = int(row["rank"])
            required = [row.get("runtime_seconds"), row.get("final_loss"), row.get("fid")]
            if any(pd.isna(v) or v == "" for v in required):
                failed_ranks.append(rank)
            else:
                completed_ranks.append(rank)
            if not pd.isna(row.get("final_loss")) and float(row.get("final_loss")) > 5.0:
                unstable_ranks.append(rank)

        viable = df.dropna(subset=["fid", "runtime_seconds"])
        if not viable.empty:
            viable = viable.copy()
            viable["score"] = viable["fid"] + 0.001 * viable["runtime_seconds"]
            best_row = viable.sort_values("score").iloc[0]
            best_rank = int(best_row["rank"])

    summary_table = "| rank | total_params | trainable_params | percent_trainable | runtime_seconds | avg_epoch_time | peak_gpu_memory_mb | final_loss | fid |\n"
    summary_table += "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    if not df.empty:
        for _, row in df.sort_values("rank").iterrows():
            summary_table += (
                f"| {int(row['rank'])} | {row.get('total_params','')} | {row.get('trainable_params','')} | "
                f"{row.get('percent_trainable','')} | {row.get('runtime_seconds','')} | {row.get('avg_epoch_time','')} | "
                f"{row.get('peak_gpu_memory_mb','')} | {row.get('final_loss','')} | {row.get('fid','')} |\n"
            )

    report_lines = [
        "# Phase 3 Report — LoRA Rank Sweep",
        "",
        f"_Auto-generated at {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Run Status",
        f"- Completed ranks: {completed_ranks if completed_ranks else 'None'}",
        f"- Failed ranks: {failed_ranks if failed_ranks else 'None'}",
        f"- Validation status: {validation.get('status', 'FAIL')}",
        "",
        "## Parameter / Runtime / Quality Table",
        summary_table,
        "",
        "## Comparative Findings",
        f"- Best quality-efficiency trade-off rank (heuristic): {best_rank}",
        f"- Unstable ranks (high final loss heuristic): {unstable_ranks if unstable_ranks else 'None observed'}",
        "",
        "## Instability / NaN / Errors",
    ]

    if validation.get("failure_reasons"):
        for reason in validation["failure_reasons"]:
            report_lines.append(f"- {reason}")
    else:
        report_lines.append("- None")

    final_status = "PASS" if validation.get("status") == "PASS" and len(failed_ranks) == 0 else "FAIL"
    report_lines.extend(["", "## Final PASS/FAIL", f"**{final_status}**", ""])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
