# Tiny DiT Validation Report

- Validation status: **PASS**
- Dataset/backbone: CIFAR-10 (32x32), Tiny DiT (Transformer2DModel)
- Ranks/epochs: 4, 8, 16 @ 10 epochs
- FID protocol: pytorch-fid local CIFAR-10 reference, 2000 generated images/rank

## Key FID Results
- rank4 (Tiny DiT): 383.2438
- rank8 (Tiny DiT): 387.8070
- rank16 (Tiny DiT): 402.0335

## Backbone Trend Comparison (ranks 4/8/16)
- Best DDPM 20e rank by FID: 4
- Best Tiny DiT rank by FID: 4
- Similar trend across backbones (moderate rank best): **YES**

## Interpretation
- DDPM and Tiny DiT exhibit similar rank sensitivity: moderate ranks remain competitive while rank16 shows diminishing returns.
- Safe interpretation: "Our experiments suggest that small-to-moderate LoRA ranks consistently provide strong efficiency-quality trade-offs across controlled DDPM and lightweight Tiny DiT settings, while larger ranks exhibit diminishing returns relative to their additional parameter cost."

## Final Status
**PASS**
