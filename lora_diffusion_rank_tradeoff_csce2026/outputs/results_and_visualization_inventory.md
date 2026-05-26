# Results and Visualization Inventory

Generated on request as an extraction/organization pass only.  
No plots were regenerated or modified during this inventory step.

## Scope inspected

- `outputs/`
- `outputs/metrics/`
- `outputs/figures/`
- `outputs/logs/`
- `scripts/`
- `paper/`

---

## 1) Metric files (CSV/JSON/TXT/LOG) containing experiment/evaluation data

### A. Core aggregated metric CSVs (best summary tables)

| File | Path | Experiment(s) | Important columns |
|---|---|---|---|
| `rank_sweep_results.csv` | `outputs/metrics/rank_sweep_results.csv` | Main DDPM rank sweep (2/4/8/16/32) | `rank,total_params,trainable_params,percent_trainable,runtime_seconds,avg_epoch_time,peak_gpu_memory_mb,final_loss,fid` |
| `extended_budget_results.csv` | `outputs/metrics/extended_budget_results.csv` | Extended-budget DDPM (4/8/16 @ 20 epochs) | `rank,epochs,total_params,trainable_params,percent_trainable,runtime_seconds,avg_epoch_time,peak_gpu_memory_mb,final_loss,fid` |
| `tiny_dit_results.csv` | `outputs/metrics/tiny_dit_results.csv` | Tiny DiT validation (4/8/16 @ 10 epochs) | `rank,epochs,total_params,trainable_params,percent_trainable,runtime_seconds,avg_epoch_time,peak_gpu_memory_mb,final_loss,fid` |

### B. FID result CSVs (per-track FID protocol outputs)

| File | Path | Experiment(s) | Important columns/notes |
|---|---|---|---|
| `fid_results.csv` | `outputs/metrics/fid_results.csv` | Main sweep + historical FID attempts | `rank,fid,num_eval_images,protocol,status,error_stage,error,eval_dir,timestamp_utc`; includes both PASS `pytorch-fid` rows and older FAIL `clean-fid` rows |
| `extended_budget_fid_results.csv` | `outputs/metrics/extended_budget_fid_results.csv` | Extended-budget DDPM | same schema as above; PASS rows for ranks 4/8/16 with `num_eval_images=2000` |
| `tiny_dit_fid_results.csv` | `outputs/metrics/tiny_dit_fid_results.csv` | Tiny DiT validation | same schema as above; PASS rows for ranks 4/8/16 with `num_eval_images=2000` |

### C. Validation/summary JSONs (artifact completeness + PASS/FAIL)

| File | Path | Experiment(s) | Key fields |
|---|---|---|---|
| `phase3_validation.json` | `outputs/logs/phase3_validation.json` | Main phase3 outputs validation | `status,checks[],failure_reasons[]` |
| `extended_budget_validation.json` | `outputs/logs/extended_budget_validation.json` | Extended-budget validation | `status,checks[],failure_reasons[]` |
| `tiny_dit_validation.json` | `outputs/logs/tiny_dit_validation.json` | Tiny DiT validation | `status,checks[],failure_reasons[]` |
| `rank16_strengthening_summary.json` | `outputs/logs/rank16_strengthening_summary.json` | Rank16 strengthening cross-track summary | `extended_budget_status,tiny_dit_status,overall_status,rank16_ddpm_10epoch_fid,rank16_ddpm_20epoch_fid,rank16_tiny_dit_10epoch_fid,...` |
| `fid_verification_result.json` | `outputs/fid_verification/fid_verification_result.json` | FID pipeline gate check | `selected_rank,num_generated,pytorch_fid_import_status,reference_status,fid_value,final_status` |
| `phase3_state_inspection.json` | `outputs/logs/phase3_state_inspection.json` | Phase3 artifact inspection | per-rank artifact status fields |

### D. Per-rank training/parameter JSON artifacts

| File family | Example path | Experiment(s) | Key fields |
|---|---|---|---|
| `training_metrics.json` | `outputs/extended_budget/rank4_20epoch/training_metrics.json` | Extended-budget, Tiny DiT, Phase3 rank dirs | `train_exit_code,runtime_seconds,steps_completed,samples_processed_estimate,samples_per_second_estimate,loss_history,loss_min,loss_max,loss_last,max_gpu_memory_mb_observed` |
| `lora_parameter_report.json` | `outputs/tiny_dit/rank4/lora_parameter_report.json` | Extended-budget, Tiny DiT, Phase3 rank dirs | `model,total_parameters,trainable_parameters,percent_trainable,target_modules,rank,alpha,dropout,timestamp_utc` |
| `run_metadata.json` | `outputs/tiny_dit/rank4/run_metadata.json` | Extended-budget, Tiny DiT, Phase3 rank dirs | environment/runtime metadata (`hostname,gpu_name,versions,dataset,hyperparameters,...`) |
| `*_fid.json` | `outputs/logs/extended_budget/rank4_20epoch_fid.json` | Track-specific FID runs | includes run config + final row payload |

### E. TXT metric-related artifacts

| File family | Example path | Purpose / metric relevance |
|---|---|---|
| `lora_freeze_check.txt` | `outputs/extended_budget/rank4_20epoch/lora_freeze_check.txt` | Verifies only LoRA params are trainable; includes `FREEZE_CHECK_STATUS` |
| `lora_candidate_modules.txt` | `outputs/extended_budget/rank4_20epoch/lora_candidate_modules.txt` | Captures discovered LoRA injection module names |

### F. Metric-bearing log files (selected high-value logs)

| File | Path | Experiment(s) | Notes |
|---|---|---|---|
| `phase3_localfid_recovery.log` | `outputs/logs/phase3_localfid_recovery.log` | Main phase3 recovery workflow | pipeline stages + PASS checks |
| `extended_budget.log` | `outputs/logs/extended_budget.log` | Extended-budget run | rank-by-rank execution summary |
| `tiny_dit_validation.log` | `outputs/logs/tiny_dit_validation.log` | Tiny DiT run | rank-by-rank execution summary |
| `extended_budget_47818614.log` | `outputs/logs/extended_budget_47818614.log` | SLURM job output | training/FID/aggregation flow for extended-budget |
| `tiny_dit_validation_47819313.log` | `outputs/logs/tiny_dit_validation_47819313.log` | SLURM job output | final successful tiny-dit job output |
| `outputs/logs/extended_budget/*.log` | directory | Per-rank DDPM extended-budget logs | detailed train/FID logs for ranks 4/8/16 |
| `outputs/logs/tiny_dit/*.log` | directory | Per-rank Tiny DiT logs | detailed train/FID logs for ranks 4/8/16 |
| `outputs/logs/rank_sweep/*.log` | directory | Main sweep rank logs | detailed phase3 rank logs |

---

## 2) Figure files inventory

Existing figures in `outputs/figures`:

| Figure | Path | What it visualizes | Paper usage | Generated by |
|---|---|---|---|---|
| `fid_vs_rank.png` | `outputs/figures/fid_vs_rank.png` | Main DDPM FID vs rank | Used in Results | `scripts/generate_phase3_figures.py` |
| `fid_vs_trainable_params.png` | `outputs/figures/fid_vs_trainable_params.png` | Main DDPM FID vs trainable params | Used in Results | `scripts/generate_phase3_figures.py` |
| `ddpm_fid_10_vs_20_epoch.png` | `outputs/figures/ddpm_fid_10_vs_20_epoch.png` | DDPM 10 vs 20 epoch FID (4/8/16) | Used in Extended-Budget Validation | `scripts/generate_extended_budget_figures.py` |
| `ddpm_vs_tiny_dit_fid_rank4_8_16.png` | `outputs/figures/ddpm_vs_tiny_dit_fid_rank4_8_16.png` | Backbone comparison DDPM vs Tiny DiT | Used in Tiny DiT Validation | `scripts/generate_tiny_dit_figures.py` |
| `runtime_vs_rank.png` | `outputs/figures/runtime_vs_rank.png` | Main DDPM runtime vs rank | Currently unused in paper tex includes | `scripts/generate_phase3_figures.py` |
| `gpu_memory_vs_rank.png` | `outputs/figures/gpu_memory_vs_rank.png` | Main DDPM GPU memory vs rank | Currently unused | `scripts/generate_phase3_figures.py` |
| `training_loss_curves.png` | `outputs/figures/training_loss_curves.png` | Main DDPM loss curves by rank | Currently unused | `scripts/generate_phase3_figures.py` |
| `rank_sample_comparison.png` | `outputs/figures/rank_sample_comparison.png` | Main DDPM rank sample grid | Currently unused | `scripts/generate_phase3_figures.py` |
| `extended_budget_fid_vs_rank.png` | `outputs/figures/extended_budget_fid_vs_rank.png` | Extended-budget FID vs rank | Currently unused in paper tex includes | `scripts/generate_extended_budget_figures.py` |
| `extended_budget_runtime_vs_rank.png` | `outputs/figures/extended_budget_runtime_vs_rank.png` | Extended-budget runtime vs rank | Currently unused | `scripts/generate_extended_budget_figures.py` |
| `extended_budget_training_loss_curves.png` | `outputs/figures/extended_budget_training_loss_curves.png` | Extended-budget loss curves | Currently unused | `scripts/generate_extended_budget_figures.py` |
| `extended_budget_rank_sample_comparison.png` | `outputs/figures/extended_budget_rank_sample_comparison.png` | Extended-budget sample grid | Currently unused | `scripts/generate_extended_budget_figures.py` |
| `tiny_dit_fid_vs_rank.png` | `outputs/figures/tiny_dit_fid_vs_rank.png` | Tiny DiT FID vs rank | Currently unused in paper tex includes | `scripts/generate_tiny_dit_figures.py` |
| `tiny_dit_runtime_vs_rank.png` | `outputs/figures/tiny_dit_runtime_vs_rank.png` | Tiny DiT runtime vs rank | Currently unused | `scripts/generate_tiny_dit_figures.py` |
| `tiny_dit_training_loss_curves.png` | `outputs/figures/tiny_dit_training_loss_curves.png` | Tiny DiT loss curves | Currently unused | `scripts/generate_tiny_dit_figures.py` |
| `tiny_dit_rank_sample_comparison.png` | `outputs/figures/tiny_dit_rank_sample_comparison.png` | Tiny DiT sample grid | Currently unused | `scripts/generate_tiny_dit_figures.py` |

---

## 3) Plotting and aggregation scripts

### Plot generators

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `scripts/generate_phase3_figures.py` | Main phase3 figure generation | `outputs/metrics/rank_sweep_results.csv`, rank configs, per-rank `training_metrics.json`, rank sample PNGs | `fid_vs_rank.png`, `fid_vs_trainable_params.png`, `runtime_vs_rank.png`, `gpu_memory_vs_rank.png`, `training_loss_curves.png`, `rank_sample_comparison.png` |
| `scripts/generate_extended_budget_figures.py` | Extended-budget DDPM figures | `outputs/metrics/extended_budget_results.csv`, `outputs/metrics/rank_sweep_results.csv`, extended-budget configs + metrics | `extended_budget_fid_vs_rank.png`, `ddpm_fid_10_vs_20_epoch.png`, `extended_budget_runtime_vs_rank.png`, `extended_budget_training_loss_curves.png`, `extended_budget_rank_sample_comparison.png` |
| `scripts/generate_tiny_dit_figures.py` | Tiny DiT + backbone comparison figures | `outputs/metrics/tiny_dit_results.csv`, `outputs/metrics/extended_budget_results.csv`, tiny configs + metrics | `tiny_dit_fid_vs_rank.png`, `ddpm_vs_tiny_dit_fid_rank4_8_16.png`, `tiny_dit_runtime_vs_rank.png`, `tiny_dit_training_loss_curves.png`, `tiny_dit_rank_sample_comparison.png` |

### Metric aggregation scripts (feed plotting)

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `scripts/aggregate_phase3_results.py` | Build main sweep aggregate table | rank configs + per-rank `training_metrics.json`, `lora_parameter_report.json`, `outputs/metrics/fid_results.csv` | `outputs/metrics/rank_sweep_results.csv` |
| `scripts/aggregate_extended_budget.py` | Build extended-budget aggregate table | extended-budget configs + per-rank metrics/params + `outputs/metrics/extended_budget_fid_results.csv` | `outputs/metrics/extended_budget_results.csv` |
| `scripts/aggregate_tiny_dit_results.py` | Build tiny-dit aggregate table | tiny configs + per-rank metrics/params + `outputs/metrics/tiny_dit_fid_results.csv` | `outputs/metrics/tiny_dit_results.csv` |

### Visualization workflow drivers

| Script | Purpose | Notes |
|---|---|---|
| `scripts/run_extended_budget.sh` | End-to-end run: train ranks, compute FID, aggregate, generate figures, validate | Produces logs in `outputs/logs/extended_budget*` and figure/metric updates |
| `scripts/run_tiny_dit_validation.sh` | End-to-end tiny-dit run + FID + aggregate + figures + validate | Produces logs in `outputs/logs/tiny_dit*` and figure/metric updates |
| `scripts/validate_phase3_outputs.py` / `validate_extended_budget.py` / `validate_tiny_dit_outputs.py` | Artifact existence and consistency checks | Useful for inventory confidence and downstream CI |

---

## 4) Generated sample image directories

Counts are based on current on-disk PNG counts.

| Experiment type | Backbone | Rank | Directory | Approx. images | Used in paper directly? |
|---|---|---:|---|---:|---|
| Main sweep (phase3) | DDPM U-Net | 2 | `outputs/phase3/rank2/generated_samples` | 1 | Indirect (may feed sample-comparison figure) |
| Main sweep (phase3) | DDPM U-Net | 2 | `outputs/phase3/rank2/eval_samples_2k` | 2000 | Indirect (FID source) |
| Main sweep (phase3) | DDPM U-Net | 4 | `outputs/phase3/rank4/generated_samples` | 1 | Indirect |
| Main sweep (phase3) | DDPM U-Net | 4 | `outputs/phase3/rank4/eval_samples_10k` | 10000 | Legacy/high-volume eval cache |
| Main sweep (phase3) | DDPM U-Net | 8 | `outputs/phase3/rank8/generated_samples` | 1 | Indirect |
| Main sweep (phase3) | DDPM U-Net | 8 | `outputs/phase3/rank8/eval_samples_10k` | 10000 | Legacy/high-volume eval cache |
| Main sweep (phase3) | DDPM U-Net | 16 | `outputs/phase3/rank16/generated_samples` | 1 | Indirect |
| Main sweep (phase3) | DDPM U-Net | 16 | `outputs/phase3/rank16/eval_samples_2k` | 2000 | Indirect (FID source) |
| Main sweep (phase3) | DDPM U-Net | 16 | `outputs/phase3/rank16/eval_samples_10k` | 1576 | Incomplete legacy folder |
| Main sweep (phase3) | DDPM U-Net | 32 | `outputs/phase3/rank32/generated_samples` | 1 | Indirect |
| Main sweep (phase3) | DDPM U-Net | 32 | `outputs/phase3/rank32/eval_samples_2k` | 2000 | Indirect (FID source) |
| Extended-budget | DDPM U-Net | 4 | `outputs/extended_budget/rank4_20epoch/generated_samples` | 1 | Indirect |
| Extended-budget | DDPM U-Net | 4 | `outputs/extended_budget/rank4_20epoch/eval_samples_2k` | 2000 | Indirect (FID source) |
| Extended-budget | DDPM U-Net | 8 | `outputs/extended_budget/rank8_20epoch/generated_samples` | 1 | Indirect |
| Extended-budget | DDPM U-Net | 8 | `outputs/extended_budget/rank8_20epoch/eval_samples_2k` | 2000 | Indirect (FID source) |
| Extended-budget | DDPM U-Net | 16 | `outputs/extended_budget/rank16_20epoch/generated_samples` | 1 | Indirect |
| Extended-budget | DDPM U-Net | 16 | `outputs/extended_budget/rank16_20epoch/eval_samples_2k` | 2000 | Indirect (FID source) |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 4 | `outputs/tiny_dit/rank4/generated_samples` | 1 | Indirect |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 4 | `outputs/tiny_dit/rank4/eval_samples_2k` | 2000 | Indirect (FID source) |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 8 | `outputs/tiny_dit/rank8/generated_samples` | 1 | Indirect |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 8 | `outputs/tiny_dit/rank8/eval_samples_2k` | 2000 | Indirect (FID source) |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 16 | `outputs/tiny_dit/rank16/generated_samples` | 1 | Indirect |
| Tiny DiT validation | Tiny DiT (`Transformer2DModel`) | 16 | `outputs/tiny_dit/rank16/eval_samples_2k` | 2000 | Indirect (FID source) |

---

## 5) Best files for future visualization (recommended sources)

### Best primary plotting inputs

1. `outputs/metrics/rank_sweep_results.csv`  
   - Clean, aggregated main sweep with FID + efficiency + systems metrics.
2. `outputs/metrics/extended_budget_results.csv`  
   - Clean aggregate for 20-epoch DDPM validation.
3. `outputs/metrics/tiny_dit_results.csv`  
   - Clean aggregate for Tiny DiT validation.

### Best FID protocol/audit sources

4. `outputs/metrics/fid_results.csv`  
   - Full historical FID record (PASS/FAIL across clean-fid and pytorch-fid).
5. `outputs/metrics/extended_budget_fid_results.csv`
6. `outputs/metrics/tiny_dit_fid_results.csv`

### Best validation/meta files for figure captions and sanity checks

7. `outputs/logs/phase3_validation.json`
8. `outputs/logs/extended_budget_validation.json`
9. `outputs/logs/tiny_dit_validation.json`
10. `outputs/logs/rank16_strengthening_summary.json`

### Additional useful context files

11. Per-rank `training_metrics.json` files (for loss curves and runtime/memory deep dives)
12. Per-rank `lora_parameter_report.json` files (trainable parameter accounting)
13. `outputs/fid_verification/fid_verification_result.json` (pipeline gating provenance)

---

## 6) Paper figure references audit (`paper/main.tex`)

Figure files currently referenced in paper LaTeX:

- `../outputs/figures/fid_vs_rank.png`
- `../outputs/figures/fid_vs_trainable_params.png`
- `../outputs/figures/ddpm_fid_10_vs_20_epoch.png`
- `../outputs/figures/ddpm_vs_tiny_dit_fid_rank4_8_16.png`

### Used vs unused (relative to existing `outputs/figures`)

**Used in paper:**
- `fid_vs_rank.png`
- `fid_vs_trainable_params.png`
- `ddpm_fid_10_vs_20_epoch.png`
- `ddpm_vs_tiny_dit_fid_rank4_8_16.png`

**Currently unused in paper text:**
- `runtime_vs_rank.png`
- `gpu_memory_vs_rank.png`
- `training_loss_curves.png`
- `rank_sample_comparison.png`
- `extended_budget_fid_vs_rank.png`
- `extended_budget_runtime_vs_rank.png`
- `extended_budget_training_loss_curves.png`
- `extended_budget_rank_sample_comparison.png`
- `tiny_dit_fid_vs_rank.png`
- `tiny_dit_runtime_vs_rank.png`
- `tiny_dit_training_loss_curves.png`
- `tiny_dit_rank_sample_comparison.png`

---

## Notes for next phase (figure redesign readiness)

- The three aggregated CSVs in `outputs/metrics` are the cleanest single-source tables for publication-quality redesign.
- For strict comparability, prefer rows with `status=PASS` and `num_eval_images=2000` in FID CSVs.
- Keep legacy `clean-fid` fail rows in `fid_results.csv` for audit history, but filter them out when plotting final paper figures.
