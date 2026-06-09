# Multi-Viscosity Burgers Hybrid Solver

This repository contains code and a pretrained checkpoint for evaluating a learned hybrid solver on forced 1D Burgers equations across multiple viscosity values.

## Contents

- `code/run_multinu_scaled_substeps_deterministic.py`: main deterministic multi-viscosity evaluation script.
- `code/train.py`: physics kernels and hybrid forward rollout used by the learned solver.
- `code/model/cp_model.py`: CP-factorized FNO/UNet correction model used by the released checkpoint.
- `code/model/conv_fno_model.py`: auxiliary model definition kept for import compatibility with `train.py`.
- `weights/model_fixed.pkl`: pretrained model parameters.
- `outputs/`: previously generated evaluation results.

The included `outputs/` directory contains the previously generated CSV/NPZ evaluation artifacts. New reruns are written to `outputs_rerun/` by default and are ignored by Git.

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

For GPU evaluation, install a CUDA-compatible `jax`/`jaxlib` build that matches your CUDA version. The evaluation script requires a JAX GPU backend and exits if only CPU devices are available.

## Run Evaluation

From the repository root:

```bash
python code/run_multinu_scaled_substeps_deterministic.py
```

By default, outputs are written to `outputs_rerun/`.

The full default run evaluates:

- native resolutions `N = 200, 400, 800, 1000, 2000`
- viscosities `nu = 0.0, 0.001, 0.005, 0.01`
- seeds `1816` through `1865`
- WENO-Z reference resolution `N = 4000`

This can be computationally expensive. For a smoke test, reduce `SEED_RANGE`, `NU_LIST`, and/or `N_NATIVE_LIST` near the top of `code/run_multinu_scaled_substeps_deterministic.py`.

## Checkpoint

The script looks for the pretrained checkpoint in:

1. `weights/model_fixed.pkl`
2. `code/model_fixed.pkl`

The released layout uses `weights/model_fixed.pkl`.
