# PALRC Hybrid Solver

This repository contains reproducibility materials for the paper:

**A Coarse-Evolution/Fine-Correction Hybrid Solver for Accelerated Flow Computations**

The paper proposes a decoupled coarse-evolution/fine-correction hybrid solver
for time-dependent flow simulation. A classical numerical solver is retained
for coarse-grid evolution, while PALRC provides lightweight target-resolution
residual correction.

## Benchmarks

The repository is organized by benchmark problem:

- `Burgers/`: 1D Burgers equation with shock and sharp-gradient diagnostics.
- `Kolmogorov_Flow/`: 2D periodic Kolmogorov flow with rollout and HCT diagnostics.
- `Lid_Driven_Cavity/`: non-periodic MAC-grid cavity flow diagnostics.

## Current release

All experimental data, corresponding analysis scripts, and trained model
weights have been fully released. The materials needed to reproduce the
reported analysis for the Burgers, Kolmogorov-flow, and lid-driven-cavity
experiments are available in this repository and the accompanying Google
Drive folder:

https://drive.google.com/drive/folders/1Zj4vO_sCJQmIEN2RZtiI1ustbW3RxbxZ?usp=sharing


## Citation

Citation information will be added.
