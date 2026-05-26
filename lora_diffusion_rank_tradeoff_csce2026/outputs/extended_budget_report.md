# Extended-Budget DDPM Report

- Validation status: **PASS**
- Dataset/backbone: CIFAR-10 (32x32), DDPM U-Net
- Ranks/epochs: 4, 8, 16 @ 20 epochs
- FID protocol: pytorch-fid local CIFAR-10 reference, 2000 generated images/rank

## Key FID Results
- rank4 (20e): 130.8580
- rank8 (20e): 132.2700
- rank16 (20e): 131.3555

## Rank16 Change vs DDPM 10 Epochs
- rank16 DDPM 10e FID: 132.2549
- rank16 DDPM 20e FID: 131.3555
- Absolute improvement (10e - 20e): 0.8994
- Relative improvement: 0.6801%
- Substantial rank16 improvement (>=5%): **NO**

## Interpretation
- Rank16 does not improve enough to overturn the efficiency-oriented moderate-rank conclusion.
- Cautious claim: small-to-moderate ranks continue to provide strong efficiency-quality trade-offs in this controlled DDPM setting.

## Final Status
**PASS**
