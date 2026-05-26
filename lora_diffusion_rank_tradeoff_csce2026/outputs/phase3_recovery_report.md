# Phase 3 Recovery Report (FID-Gated)

- Timestamp (UTC): 2026-05-25T22:09:10+00:00
- FID protocol: `pytorch-fid` local-folder reference
- FID verification status: **PASS**
- Selected verification rank: `rank2`
- Test FID (100 generated images): `221.735974`
- Full recovery continued: **YES**

## FID Verification Diagnostic

- `pytorch_fid` import: PASS
- Local reference folder: `outputs/fid_reference/cifar10_test/` (10k PNGs, RGB, 32x32)
- Verification generated images: `outputs/fid_verification/rank4/generated_100/`
- Gate status source: `outputs/fid_verification/fid_verification_result.json`
- Output files:
  - `outputs/fid_verification/fid_verification_result.json`
  - `outputs/fid_verification/fid_verification_report.md`

## Recovery Results

- Completed ranks currently on disk: `2, 4, 8, 16, 32`
- Recovered FIDs (2k protocol):
  - rank2: `131.415373`
  - rank4: `124.138043`
  - rank8: `124.213620`
  - rank16: `132.254947`
  - rank32: `131.647877`
- Rank32 retraining needed: **NO** (recovery completed)
- Aggregation generated: `outputs/metrics/rank_sweep_results.csv`
- Figures generated:
  - `outputs/figures/fid_vs_rank.png`
  - `outputs/figures/fid_vs_trainable_params.png`
  - `outputs/figures/runtime_vs_rank.png`
  - `outputs/figures/gpu_memory_vs_rank.png`
  - `outputs/figures/training_loss_curves.png`
  - `outputs/figures/rank_sample_comparison.png`
- Validation result: **PASS** (`outputs/logs/phase3_validation.json`)

## Final PASS/FAIL

**PASS**

## Next Recommended Action

1. Optional: clean up stale legacy `clean-fid` fail rows (10k protocol) in `outputs/metrics/fid_results.csv` if you want a pure local-protocol table.
2. Keep `slurm/recover_rank32.slurm` for reproducibility or future reruns.
3. Proceed to paper analysis using `outputs/metrics/rank_sweep_results.csv` and generated figures.
