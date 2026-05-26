# Phase 3 Report — LoRA Rank Sweep

_Auto-generated at 2026-05-25T22:08:46.840446+00:00_

## Run Status
- Completed ranks: [2, 4, 8, 16, 32]
- Failed ranks: None
- Validation status: PASS

## Parameter / Runtime / Quality Table
| rank | total_params | trainable_params | percent_trainable | runtime_seconds | avg_epoch_time | peak_gpu_memory_mb | final_loss | fid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 113722371.0 | 49152.0 | 0.043221 | 521.0 | 52.1 | 2048.21 | 0.0532174408435821 | 131.415373 |
| 4 | 113771523.0 | 98304.0 | 0.086405 | 522.0 | 52.2 | 2048.89 | 0.0594035983085632 | 124.138043 |
| 8 | 113869827.0 | 196608.0 | 0.17266 | 524.0 | 52.4 | 2050.24 | 0.0586773753166198 | 124.21362 |
| 16 | 114066435.0 | 393216.0 | 0.344725 | 521.0 | 52.1 | 2052.95 | 0.0413772761821746 | 132.254947 |
| 32 | 114459651.0 | 786432.0 | 0.687082 | 528.0 | 52.8 | 2074.49 | 0.0633558854460716 | 131.647877 |


## Comparative Findings
- Best quality-efficiency trade-off rank (heuristic): 4
- Unstable ranks (high final loss heuristic): None observed

## Instability / NaN / Errors
- None

## Final PASS/FAIL
**PASS**
