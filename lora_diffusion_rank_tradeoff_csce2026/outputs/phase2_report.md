# Phase 2 Report — LoRA DDPM Smoke Validation

_Auto-generated at 2026-05-24T20:54:41.729857+00:00_

## LoRA Injection
- Injection succeeded: True
- Target modules requested: ['to_q', 'to_k', 'to_v', 'to_out.0']
- Trainable LoRA layers found: 48
- Layers with trainable LoRA parameters:
  - `base_model.model.down_blocks.4.attentions.0.to_k.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_k.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_out.0.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_out.0.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_q.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_q.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_v.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.0.to_v.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_k.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_k.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_out.0.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_out.0.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_q.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_q.lora_B.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_v.lora_A.default.weight`
  - `base_model.model.down_blocks.4.attentions.1.to_v.lora_B.default.weight`
  - `base_model.model.mid_block.attentions.0.to_k.lora_A.default.weight`
  - `base_model.model.mid_block.attentions.0.to_k.lora_B.default.weight`
  - `base_model.model.mid_block.attentions.0.to_out.0.lora_A.default.weight`
  - `base_model.model.mid_block.attentions.0.to_out.0.lora_B.default.weight`
  - `base_model.model.mid_block.attentions.0.to_q.lora_A.default.weight`
  - `base_model.model.mid_block.attentions.0.to_q.lora_B.default.weight`
  - `base_model.model.mid_block.attentions.0.to_v.lora_A.default.weight`
  - `base_model.model.mid_block.attentions.0.to_v.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_k.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_k.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_out.0.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_out.0.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_q.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_q.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_v.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.0.to_v.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_k.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_k.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_out.0.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_out.0.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_q.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_q.lora_B.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_v.lora_A.default.weight`
  - `base_model.model.up_blocks.1.attentions.1.to_v.lora_B.default.weight`

## Parameter Statistics
- Total parameters: 113771523
- Trainable parameters: 98304
- Percent trainable: 0.086405

## Freeze Verification
- Backbone freezing succeeded: True
- Freeze report path: `/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/logs/lora_freeze_check.txt`

## Training
- Training command: `python scripts/train_lora_ddpm.py --config configs/lora_smoke.yaml`
- Training completed: True
- Runtime seconds: 52

## Artifacts
- Output directory: `/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/smoke/lora_rank4_smoke`
- Sample image paths: ['/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/smoke/lora_rank4_smoke/generated_samples/sample_000.png']
- LoRA checkpoint path: `/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/smoke/lora_rank4_smoke/lora_adapter`
- Merged model path: `/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/smoke/lora_rank4_smoke/unet`
- Scheduler config path: `/users/PCS0229/imankhazrak/lora-diffusion-rank-tradeoffs/lora_diffusion_rank_tradeoff_csce2026/outputs/smoke/lora_rank4_smoke/scheduler/scheduler_config.json`

## Validation
- Validation status: PASS
- Warnings/Errors: None

## Final Status
**PASS**
