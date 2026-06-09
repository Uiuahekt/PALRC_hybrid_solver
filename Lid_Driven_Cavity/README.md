Cavity-flow evaluation release
==============================

This folder contains the GitHub release version of the cavity-flow evaluation
files. The package is organized by method and by evaluation metric.

Data download
-------------

The large NetCDF data files are not stored in this GitHub folder. Download the
data from:

https://drive.google.com/drive/folders/1Zj4vO_sCJQmIEN2RZtiI1ustbW3RxbxZ?usp=sharing

After downloading, place each data file in the corresponding method's `data/`
folder:

- Ours:
  `methods/Ours_palrc_full_eval_MAC_32bit_epoch84/data/full_trajectories_MAC_32bit.nc`
- FNO:
  `methods/FNO_rollout_30cases_400steps/data/fno_predictions_30cases_20steps.nc`
- GNOT:
  `methods/GNOT_rollout_30cases_400steps/data/gnot_predictions_30cases_400steps.nc`

Directory layout
----------------

- `methods/Ours_palrc_full_eval_MAC_32bit_epoch84/`
  - Our method, corresponding to the original `palrc_full_eval_MAC_32bit_epoch84`.
- `methods/FNO_rollout_30cases_400steps/`
  - FNO baseline.
- `methods/GNOT_rollout_30cases_400steps/`
  - GNOT baseline.

Each method folder contains:

- `data/`
  - Placeholder for the required NetCDF file.
- `loss_calculation/`
  - Metric-specific scripts for recomputing losses.
- `results/`
  - Reported result files grouped by metric.

Loss calculation scripts
------------------------

Each method uses the same metric-oriented script names:

- `cell_center_velocity_error.py`
  - Computes relative L2 and max velocity errors on the cell-centered grid.
  - Writes to `../results/cell_center_velocity_error/`.
- `centerline_rms_l2_error.py`
  - Computes centerline RMS/L2 error diagnostics.
  - Writes to `../results/center_line_error/`.
- `corner_vorticity_error.py`
  - Computes localized bottom-corner vorticity errors.
  - Writes to `../results/corner_vorticity/`.
- `group_cd_metrics_no_vortex.py`
  - Computes grouped cavity-flow diagnostics without secondary-vortex metrics.
  - Writes to `../results/group_cd_metrics_no_vortex/`.

Dependencies
------------

```bash
pip install -r requirements.txt
```

Example
-------

```bash
cd methods/Ours_palrc_full_eval_MAC_32bit_epoch84/loss_calculation
python centerline_rms_l2_error.py
```

Run scripts from inside each method's `loss_calculation/` directory so the
relative `../data/` and `../results/` paths resolve correctly.
