"""
Group C/D kinetic-energy and enstrophy loss export.

This script computes only the following two diagnostics:
  1. Kinetic energy
  2. Total enstrophy

Secondary-vortex / corner-vorticity diagnostics are removed.

Aggregation order:
  1. For each held-out trajectory and each time step, compute the diagnostic
     for the prediction and for its own reference trajectory.
  2. Convert that pair into a per-trajectory loss.
  3. Aggregate the loss over trajectories at each time step.
  4. For scalar summaries, first average each trajectory over time, then
     aggregate over trajectories.

Input NetCDF is assumed to store MAC/staggered velocity fields:
    u: (num_traj, num_steps, Nx+1, Ny)
    v: (num_traj, num_steps, Nx,   Ny+1)

Expected variables:
    u_ref, v_ref
    u_ai, v_ai
    u_dns128, v_dns128
    u_dns64, v_dns64

Outputs:
    group_cd_loss_timeseries.csv
    group_cd_raw_diagnostics.csv
    group_cd_per_traj_step_losses.csv
    group_cd_scalar_summary.csv
    group_cd_scalar_summary.txt
    group_cd_scalar_summary.tex
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# User settings: only edit these two paths
# ============================================================

NC_FILE = "../data/full_trajectories_MAC_32bit.nc"
OUTPUT_DIR = "../results/group_cd_metrics_no_vortex"


# ============================================================
# Configuration
# ============================================================

L = 1.0
DT_SAVE = 0.1
COMPUTE_DTYPE = np.float32
EPS = 1e-12

MODEL_FIELD_MAP: Dict[str, Tuple[str, str]] = {
    "ref": ("u_ref", "v_ref"),
    "Ours": ("u_ai", "v_ai"),
    "dns128": ("u_dns128", "v_dns128"),
    "dns64": ("u_dns64", "v_dns64"),
    # If available, add for example:
    # "FNO": ("u_fno", "v_fno"),
    # "GNOT": ("u_gnot", "v_gnot"),
}

EVAL_METHODS = [m for m in MODEL_FIELD_MAP if m != "ref"]

METHOD_ORDER = ["dns64", "dns128", "Ours"]

METHOD_LABELS_TEX = {
    "Ours": r"Ours",
    "dns128": r"DNS\_128$\times$128",
    "dns64": r"DNS\_64$\times$64",
    "FNO": r"FNO",
    "GNOT": r"GNOT",
}


# ============================================================
# MAC / CC utilities
# ============================================================

def validate_mac_pair(u: np.ndarray, v: np.ndarray, name: str) -> None:
    """
    Validate MAC-grid shape:
        u: (..., Nx+1, Ny)
        v: (..., Nx, Ny+1)
    """
    if u.ndim != v.ndim:
        raise ValueError(
            f"[{name}] u and v have different ndim: "
            f"u.ndim={u.ndim}, v.ndim={v.ndim}"
        )

    if u.ndim < 3:
        raise ValueError(
            f"[{name}] expected at least 3D arrays (..., Nx+1, Ny), "
            f"but got u.shape={u.shape}, v.shape={v.shape}"
        )

    if u.shape[:-2] != v.shape[:-2]:
        raise ValueError(
            f"[{name}] leading dimensions do not match:\n"
            f"  u.shape={u.shape}\n"
            f"  v.shape={v.shape}"
        )

    nx = v.shape[-2]
    ny = u.shape[-1]

    expected_u_last2 = (nx + 1, ny)
    expected_v_last2 = (nx, ny + 1)

    if u.shape[-2:] != expected_u_last2 or v.shape[-2:] != expected_v_last2:
        raise ValueError(
            f"[{name}] invalid MAC-grid pair.\n"
            f"  Expected u last dims: {expected_u_last2} = (Nx+1, Ny)\n"
            f"  Expected v last dims: {expected_v_last2} = (Nx, Ny+1)\n"
            f"  Got u.shape={u.shape}, v.shape={v.shape}"
        )


def batch_mac_to_cc(
    u_mac: np.ndarray,
    v_mac: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MAC staggered velocity to cell-centered velocity.

    Input:
        u_mac: (T, Nx+1, Ny)
        v_mac: (T, Nx, Ny+1)

    Output:
        u_cc: (T, Nx, Ny)
        v_cc: (T, Nx, Ny)
    """
    validate_mac_pair(u_mac, v_mac, "batch_mac_to_cc input")

    half = COMPUTE_DTYPE(0.5)

    u_cc = half * (u_mac[:, :-1, :] + u_mac[:, 1:, :])
    v_cc = half * (v_mac[:, :, :-1] + v_mac[:, :, 1:])

    return (
        u_cc.astype(COMPUTE_DTYPE, copy=False),
        v_cc.astype(COMPUTE_DTYPE, copy=False),
    )


def infer_grid_spacing_from_mac(
    u_mac: np.ndarray,
    v_mac: np.ndarray,
) -> Tuple[int, int, float, float]:
    """
    Infer Nx, Ny, dx, dy from MAC arrays.

    Input:
        u_mac: (T, Nx+1, Ny)
        v_mac: (T, Nx, Ny+1)
    """
    validate_mac_pair(u_mac, v_mac, "infer_grid_spacing_from_mac")

    nx = v_mac.shape[-2]
    ny = u_mac.shape[-1]

    dx = L / nx
    dy = L / ny

    return nx, ny, dx, dy


# ============================================================
# Diagnostics
# ============================================================

def compute_vorticity_cc(
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Compute scalar vorticity:
        omega = dv/dx - du/dy

    Arrays are assumed to have shape (Nx, Ny):
        axis 0: x
        axis 1: y
    """
    _, du_dy = np.gradient(u_cc, dx, dy, edge_order=2)
    dv_dx, _ = np.gradient(v_cc, dx, dy, edge_order=2)
    return dv_dx - du_dy


def total_kinetic_energy(
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """
    K(t) = 0.5 * int_Omega (u^2 + v^2) dA.
    """
    return float(0.5 * np.sum(u_cc * u_cc + v_cc * v_cc) * dx * dy)


def total_enstrophy(
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """
    Z(t) = 0.5 * int_Omega omega^2 dA.
    """
    omega = compute_vorticity_cc(u_cc, v_cc, dx, dy)
    return float(0.5 * np.sum(omega * omega) * dx * dy)


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_raw_by_time(
    values: np.ndarray,
    time_rel: np.ndarray,
    quantity: str,
    method: str,
) -> Iterable[dict]:
    """
    values: (N_traj, T)

    Output raw diagnostic mean/std/median/P95 across trajectories.
    """
    mean_t = np.nanmean(values, axis=0)
    std_t = np.nanstd(values, axis=0, ddof=0)
    median_t = np.nanpercentile(values, 50, axis=0)
    p95_t = np.nanpercentile(values, 95, axis=0)
    n_traj = values.shape[0]

    for i, t in enumerate(time_rel):
        yield {
            "time_rel": float(t),
            "quantity": quantity,
            "method": method,
            "value_mean": float(mean_t[i]),
            "value_std": float(std_t[i]),
            "value_median": float(median_t[i]),
            "value_p95": float(p95_t[i]),
            "n_traj": int(n_traj),
        }


def summarize_loss_by_time(
    losses: np.ndarray,
    time_rel: np.ndarray,
    quantity: str,
    method: str,
) -> Iterable[dict]:
    """
    losses: (N_traj, T)

    Output one row per time step with mean/std/median/P95 across trajectories.
    """
    mean_t = np.nanmean(losses, axis=0)
    std_t = np.nanstd(losses, axis=0, ddof=0)
    median_t = np.nanpercentile(losses, 50, axis=0)
    p95_t = np.nanpercentile(losses, 95, axis=0)
    n_traj = losses.shape[0]

    for i, t in enumerate(time_rel):
        yield {
            "time_rel": float(t),
            "quantity": quantity,
            "method": method,
            "loss_mean": float(mean_t[i]),
            "loss_std": float(std_t[i]),
            "loss_median": float(median_t[i]),
            "loss_p95": float(p95_t[i]),
            "n_traj": int(n_traj),
        }


def summarize_scalar_loss(
    losses: np.ndarray,
    quantity: str,
    method: str,
) -> dict:
    """
    Scalar aggregation:
        1. Average loss over time for each trajectory.
        2. Aggregate the per-trajectory time-mean losses over trajectories.
    """
    per_traj_time_mean = np.nanmean(losses, axis=1)

    return {
        "quantity": quantity,
        "method": method,
        "Averaging": "per-trajectory time-mean; ensemble statistics",
        "Value": float(np.nanmean(per_traj_time_mean)),
        "Std": float(np.nanstd(per_traj_time_mean, ddof=0)),
        "Median": float(np.nanpercentile(per_traj_time_mean, 50)),
        "P95": float(np.nanpercentile(per_traj_time_mean, 95)),
        "n_traj": int(losses.shape[0]),
        "n_steps": int(losses.shape[1]),
    }


def build_per_traj_step_loss_rows(
    losses: np.ndarray,
    time_rel: np.ndarray,
    quantity: str,
    method: str,
) -> list[dict]:
    """
    Save raw per-trajectory, per-step loss values.
    """
    n_traj, n_steps = losses.shape

    rows = []
    for traj_idx in range(n_traj):
        for step in range(n_steps):
            rows.append(
                {
                    "trajectory_index_0based": int(traj_idx),
                    "trajectory_number_1based": int(traj_idx + 1),
                    "step": int(step + 1),
                    "time_rel": float(time_rel[step]),
                    "quantity": quantity,
                    "method": method,
                    "loss": float(losses[traj_idx, step]),
                }
            )

    return rows


# ============================================================
# Output writers
# ============================================================

def write_scalar_txt(df_scalar: pd.DataFrame, path: Path) -> None:
    """
    Write scalar summary to a plain text file.
    """
    lines = []
    lines.append("Group C/D Scalar Loss Summary")
    lines.append("")
    lines.append("Included diagnostics:")
    lines.append("  1. KineticEnergyAbsLoss")
    lines.append("  2. KineticEnergyRelativeLoss")
    lines.append("  3. TotalEnstrophyAbsLoss")
    lines.append("  4. TotalEnstrophyRelativeLoss")
    lines.append("")
    lines.append("Secondary-vortex / corner-vorticity diagnostics are not computed.")
    lines.append("")
    lines.append("Aggregation order:")
    lines.append("  1. Compute diagnostic for each trajectory and time step.")
    lines.append("  2. Convert prediction/reference diagnostic pair to per-trajectory loss.")
    lines.append("  3. For scalar values, first average each trajectory over time,")
    lines.append("     then aggregate over trajectories.")
    lines.append("")
    lines.append("Columns:")
    lines.append("  Value ± Std, where Std is computed over per-trajectory time-mean losses.")
    lines.append("")
    lines.append(
        f"{'Method':<12}"
        f"{'Quantity':<32}"
        f"{'Value':<16}"
        f"{'Std':<16}"
        f"{'Median':<16}"
        f"{'P95':<16}"
    )
    lines.append("-" * 108)

    df_print = df_scalar.copy()

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    df_print["method_rank"] = df_print["method"].map(method_rank).fillna(999)

    quantity_order = {
        "KineticEnergyAbsLoss": 0,
        "KineticEnergyRelativeLoss": 1,
        "TotalEnstrophyAbsLoss": 2,
        "TotalEnstrophyRelativeLoss": 3,
    }
    df_print["quantity_rank"] = df_print["quantity"].map(quantity_order).fillna(999)

    df_print = df_print.sort_values(["method_rank", "quantity_rank"])

    for _, row in df_print.iterrows():
        lines.append(
            f"{str(row['method']):<12}"
            f"{str(row['quantity']):<32}"
            f"{float(row['Value']):<16.6e}"
            f"{float(row['Std']):<16.6e}"
            f"{float(row['Median']):<16.6e}"
            f"{float(row['P95']):<16.6e}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(df_scalar: pd.DataFrame, path: Path) -> None:
    """
    Write a compact LaTeX table for kinetic-energy and enstrophy losses.

    The table reports:
        KineticEnergyAbsLoss
        TotalEnstrophyAbsLoss

    These are the two absolute diagnostic errors usually used in the paper table.
    """
    quantities = [
        ("KineticEnergyAbsLoss", r"$e_K$"),
        ("TotalEnstrophyAbsLoss", r"$e_Z$"),
    ]

    table = {}

    for method in METHOD_ORDER:
        table[method] = []

        for quantity, _label in quantities:
            sub = df_scalar[
                (df_scalar["method"] == method)
                & (df_scalar["quantity"] == quantity)
            ]

            if sub.empty:
                table[method].append((np.nan, np.nan))
            else:
                row = sub.iloc[0]
                table[method].append((float(row["Value"]), float(row["Std"])))

    best = []
    for j in range(len(quantities)):
        vals = [table[m][j][0] for m in table if not np.isnan(table[m][j][0])]
        best.append(min(vals) if vals else np.nan)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Kinetic-energy and enstrophy diagnostic losses.}",
        r"\label{tab:ke_enstrophy_losses}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        (
            r"Method & "
            + " & ".join(label for _, label in quantities)
            + r" \\"
        ),
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        if method not in table:
            continue

        label = METHOD_LABELS_TEX.get(method, method)
        cells = []

        for j, (mean_val, std_val) in enumerate(table[method]):
            if np.isnan(mean_val):
                text = "--"
            else:
                text = f"{mean_val:.4e} $\\pm$ {std_val:.1e}"
                if np.isclose(mean_val, best[j]):
                    text = r"\textbf{" + text + "}"
            cells.append(text)

        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main pipeline
# ============================================================

def main(
    nc_file: str = NC_FILE,
    output_dir: str = OUTPUT_DIR,
    max_trajs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nc_path = Path(nc_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.exists():
        raise FileNotFoundError(f"Input NetCDF file does not exist: {nc_path}")

    output_loss_csv = out_dir / "group_cd_loss_timeseries.csv"
    output_raw_csv = out_dir / "group_cd_raw_diagnostics.csv"
    output_per_traj_csv = out_dir / "group_cd_per_traj_step_losses.csv"
    output_scalar_csv = out_dir / "group_cd_scalar_summary.csv"
    output_scalar_txt = out_dir / "group_cd_scalar_summary.txt"
    output_tex = out_dir / "group_cd_scalar_summary.tex"

    print("=" * 72)
    print("Kinetic-energy / enstrophy diagnostic-loss pipeline")
    print("=" * 72)
    print(f"Input NetCDF: {nc_path}")
    print(f"Output dir:   {out_dir}")

    with xr.open_dataset(nc_path) as ds:
        required_vars = [
            var
            for pair in MODEL_FIELD_MAP.values()
            for var in pair
        ]

        missing = [var for var in required_vars if var not in ds.variables]
        if missing:
            raise KeyError(
                "These variables are missing from the NetCDF file:\n  "
                + "\n  ".join(missing)
            )

        ref_u_name, ref_v_name = MODEL_FIELD_MAP["ref"]
        ref_u_shape = ds[ref_u_name].shape
        ref_v_shape = ds[ref_v_name].shape

        if len(ref_u_shape) != 4 or len(ref_v_shape) != 4:
            raise ValueError(
                "Expected reference arrays to have shape "
                "(num_traj, num_steps, ..., ...).\n"
                f"Got {ref_u_name}.shape={ref_u_shape}, "
                f"{ref_v_name}.shape={ref_v_shape}"
            )

        n_traj_total = int(ref_u_shape[0])
        n_steps = int(ref_u_shape[1])
        n_traj = n_traj_total if max_trajs is None else min(max_trajs, n_traj_total)

        # Use the first trajectory to infer MAC grid and spacing.
        u_ref_0 = ds[ref_u_name].isel({ds[ref_u_name].dims[0]: 0}).values.astype(
            COMPUTE_DTYPE,
            copy=False,
        )
        v_ref_0 = ds[ref_v_name].isel({ds[ref_v_name].dims[0]: 0}).values.astype(
            COMPUTE_DTYPE,
            copy=False,
        )

        nx, ny, dx, dy = infer_grid_spacing_from_mac(u_ref_0, v_ref_0)

        time_rel = np.arange(1, n_steps + 1, dtype=np.float64) * DT_SAVE

        print(f"Trajectories: {n_traj}/{n_traj_total}")
        print(f"Steps:        {n_steps}")
        print("MAC grid:")
        print(f"  {ref_u_name}.shape = {ref_u_shape} = (traj, step, Nx+1, Ny)")
        print(f"  {ref_v_name}.shape = {ref_v_shape} = (traj, step, Nx, Ny+1)")
        print(f"  inferred CC grid = {nx} x {ny}")
        print(f"  dx = {dx:.6e}, dy = {dy:.6e}")
        print(f"Methods: {', '.join(MODEL_FIELD_MAP.keys())}")

        # Storage: diagnostic[method] -> (N_traj, T)
        ke = {
            m: np.zeros((n_traj, n_steps), dtype=np.float64)
            for m in MODEL_FIELD_MAP
        }
        enstrophy = {
            m: np.zeros((n_traj, n_steps), dtype=np.float64)
            for m in MODEL_FIELD_MAP
        }

        # ------------------------------------------------------------
        # Compute raw diagnostics per trajectory and time step
        # ------------------------------------------------------------
        for traj_idx in range(n_traj):
            cc_cache = {}

            for method, (u_name, v_name) in MODEL_FIELD_MAP.items():
                u_da = ds[u_name]
                v_da = ds[v_name]

                u_mac = u_da.isel({u_da.dims[0]: traj_idx}).values.astype(
                    COMPUTE_DTYPE,
                    copy=False,
                )
                v_mac = v_da.isel({v_da.dims[0]: traj_idx}).values.astype(
                    COMPUTE_DTYPE,
                    copy=False,
                )

                validate_mac_pair(u_mac, v_mac, method)
                cc_cache[method] = batch_mac_to_cc(u_mac, v_mac)

            for step in range(n_steps):
                for method, (u_cc_all, v_cc_all) in cc_cache.items():
                    u_cc = u_cc_all[step]
                    v_cc = v_cc_all[step]

                    ke[method][traj_idx, step] = total_kinetic_energy(
                        u_cc,
                        v_cc,
                        dx,
                        dy,
                    )

                    enstrophy[method][traj_idx, step] = total_enstrophy(
                        u_cc,
                        v_cc,
                        dx,
                        dy,
                    )

            if (traj_idx + 1) % 5 == 0 or traj_idx == n_traj - 1:
                print(f"Finished {traj_idx + 1}/{n_traj} trajectories")

    raw_rows = []
    loss_rows = []
    per_traj_rows = []
    scalar_rows = []

    # ------------------------------------------------------------
    # Raw diagnostic curves
    # ------------------------------------------------------------
    for method in MODEL_FIELD_MAP:
        raw_rows.extend(
            summarize_raw_by_time(
                ke[method],
                time_rel,
                "KineticEnergy",
                method,
            )
        )

        raw_rows.extend(
            summarize_raw_by_time(
                enstrophy[method],
                time_rel,
                "TotalEnstrophy",
                method,
            )
        )

    # ------------------------------------------------------------
    # Per-trajectory losses, then aggregate
    # ------------------------------------------------------------
    for method in EVAL_METHODS:
        # Kinetic energy losses
        ke_abs = np.abs(ke[method] - ke["ref"])
        ke_rel = ke_abs / (np.abs(ke["ref"]) + EPS)

        loss_rows.extend(
            summarize_loss_by_time(
                ke_abs,
                time_rel,
                "KineticEnergyAbsLoss",
                method,
            )
        )

        loss_rows.extend(
            summarize_loss_by_time(
                ke_rel,
                time_rel,
                "KineticEnergyRelativeLoss",
                method,
            )
        )

        scalar_rows.append(
            summarize_scalar_loss(
                ke_abs,
                "KineticEnergyAbsLoss",
                method,
            )
        )

        scalar_rows.append(
            summarize_scalar_loss(
                ke_rel,
                "KineticEnergyRelativeLoss",
                method,
            )
        )

        per_traj_rows.extend(
            build_per_traj_step_loss_rows(
                ke_abs,
                time_rel,
                "KineticEnergyAbsLoss",
                method,
            )
        )

        per_traj_rows.extend(
            build_per_traj_step_loss_rows(
                ke_rel,
                time_rel,
                "KineticEnergyRelativeLoss",
                method,
            )
        )

        # Enstrophy losses
        ens_abs = np.abs(enstrophy[method] - enstrophy["ref"])
        ens_rel = ens_abs / (np.abs(enstrophy["ref"]) + EPS)

        loss_rows.extend(
            summarize_loss_by_time(
                ens_abs,
                time_rel,
                "TotalEnstrophyAbsLoss",
                method,
            )
        )

        loss_rows.extend(
            summarize_loss_by_time(
                ens_rel,
                time_rel,
                "TotalEnstrophyRelativeLoss",
                method,
            )
        )

        scalar_rows.append(
            summarize_scalar_loss(
                ens_abs,
                "TotalEnstrophyAbsLoss",
                method,
            )
        )

        scalar_rows.append(
            summarize_scalar_loss(
                ens_rel,
                "TotalEnstrophyRelativeLoss",
                method,
            )
        )

        per_traj_rows.extend(
            build_per_traj_step_loss_rows(
                ens_abs,
                time_rel,
                "TotalEnstrophyAbsLoss",
                method,
            )
        )

        per_traj_rows.extend(
            build_per_traj_step_loss_rows(
                ens_rel,
                time_rel,
                "TotalEnstrophyRelativeLoss",
                method,
            )
        )

    df_raw = pd.DataFrame(raw_rows)
    df_loss = pd.DataFrame(loss_rows)
    df_per_traj = pd.DataFrame(per_traj_rows)
    df_scalar = pd.DataFrame(scalar_rows)

    df_raw.to_csv(output_raw_csv, index=False)
    df_loss.to_csv(output_loss_csv, index=False)
    df_per_traj.to_csv(output_per_traj_csv, index=False)
    df_scalar.to_csv(output_scalar_csv, index=False)

    write_scalar_txt(df_scalar, output_scalar_txt)
    write_latex_table(df_scalar, output_tex)

    print("\nSaved files:")
    print(f"  {output_loss_csv}")
    print(f"  {output_raw_csv}")
    print(f"  {output_per_traj_csv}")
    print(f"  {output_scalar_csv}")
    print(f"  {output_scalar_txt}")
    print(f"  {output_tex}")

    print("\nDone.")

    return df_loss, df_raw, df_scalar


if __name__ == "__main__":
    main()