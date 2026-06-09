# Absolute Error Analysis Release

This folder contains the trajectory data release and the analysis script used to reproduce the absolute-error statistics and figures. Three model trajectory datasets are currently available: `Ours_base`, `Ours_large`, and `Ours_large_proj`.

## Folder Structure

- `data/`: place the downloaded trajectory NetCDF file here.
- `analyze_all_method_trajectories.py`: analysis script.
- `history_hct_final.csv` : Raw high-correlation-time results under the HCT95 and HCT80 criteria for all evaluated methods, including baseline solvers, learned comparison methods, the proposed method, and all ablation variants.

- `output/Ours_base/`: analysis results for the `Ours_base` trajectory dataset.
- `output/Ours_large/`: analysis results for the `Ours_large` trajectory dataset.
- `output/Ours_large_proj/`: analysis results for the `Ours_large_proj` trajectory dataset.

## Download Data

The released trajectory datasets are available from Google Drive:

```text
https://drive.google.com/drive/folders/1Zj4vO_sCJQmIEN2RZtiI1ustbW3RxbxZ?usp=sharing
```

The corresponding trained model weights have also been fully released and can be found in the same Google Drive folder.

Download any one of the three model trajectory files from the folder above and use it for the corresponding analysis. For the default command below, place the downloaded file at:

```text
data/all_method_trajectories.nc
```

You can also keep the downloaded filename and pass it explicitly with `--trajectory-nc`.

## Dataset

Each trajectory NetCDF file contains:

- `u(method, seed, time, x, y)`
- `v(method, seed, time, x, y)`

Dimensions:

- `method = 8`: `baseline_64x64`, `baseline_128x128`, `baseline_256x256`, `baseline_512x512`, `baseline_1024x1024`, `baseline_2048x2048`, `LI`, `Learned_Correction`
- `seed = 32`
- `time = 199`
- `x = 64`
- `y = 64`

## HCT CSV File

This release includes the raw HCT results in:

* `history_hct_final.csv` reports the seed-wise HCT values for all evaluated methods, including the proposed method, baseline solvers, learned comparison methods, and ablation variants. Its columns are:

  * `Method`: method name.
  * `Seed`: zero-based seed index.
  * `HCT_0.95`: HCT value for that method and seed under the 95% correlation criterion.
  * `HCT_0.80`: HCT value for that method and seed under the 80% correlation criterion.

This file provides the raw per-seed HCT95 and HCT80 data before method-level averaging.

## Run Analysis

From this folder, after placing one downloaded trajectory file at `data/all_method_trajectories.nc`:

```bash
python analyze_all_method_trajectories.py
```

The script reads:

```text
data/all_method_trajectories.nc
```

and writes results to:

```text
output/
```

For the released results, the analysis has already been run for all three available model trajectories. The corresponding outputs are stored in:

```text
output/Ours_base/
output/Ours_large/
output/Ours_large_proj/
```

## Outputs

Each output subfolder contains:

- `absolute_error_all_methods_from_trajectories.nc`
- `absolute_error_summary.nc`
- `absolute_error_overall_statistics.csv`
- `absolute_error_stage_statistics.csv`
- `abs_error_vs_time_base.eps`
- `abs_error_vs_time_base.png`

The default reference method is `baseline_2048x2048`.

To specify custom paths:

```bash
python analyze_all_method_trajectories.py \
  --trajectory-nc data/Ours_base_all_method_trajectories.nc \
  --output-dir output/Ours_base
```
