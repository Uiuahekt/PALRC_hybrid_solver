"""
Compute paper-consistent bottom-corner vorticity errors from a CC-grid NetCDF file.

This script computes the localized corner-vorticity field error used in the paper.

Input NetCDF is assumed to store cell-centered / collocated velocity fields:
    u: (num_traj, num_steps, Nx, Ny)
    v: (num_traj, num_steps, Nx, Ny)

Typical FNO NetCDF variables:
    gt_u, gt_v
    pred_u, pred_v

Typical Ours/DNS NetCDF variables:
    u_ref, v_ref
    u_ai, v_ai
    u_dns128, v_dns128
    u_dns64, v_dns64

The code directly uses the CC velocity fields and computes cell-centered vorticity:
    omega = dv/dx - du/dy

Paper-consistent bottom-corner masks:
    n_c = floor(Nx / 3)

    I_BL = {(i,j): 2 <= i < n_c,        2 <= j < n_c}
    I_BR = {(i,j): Nx-n_c <= i < Nx-2, 2 <= j < n_c}

For each method, trajectory, time step, and corner:
    e_omega_s(t) =
        sqrt(sum_{(i,j) in I_s} (omega_pred - omega_ref)^2 dx dy)

    e_rel_omega_s(t) =
        e_omega_s(t) / max(sqrt(sum_{(i,j) in I_s} omega_ref^2 dx dy), EPS0)

Outputs:
    corner_vorticity_loss_timeseries.csv
    corner_vorticity_per_traj_step_losses.csv
    corner_vorticity_ref_norm_timeseries.csv
    corner_vorticity_scalar_summary.csv
    corner_vorticity_scalar_summary.txt
    corner_vorticity_scalar_summary.tex
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

NC_FILE = "../data/gnot_predictions_30cases_400steps.nc"
OUTPUT_DIR = "../results/corner_vorticity"


# ============================================================
# Configuration
# ============================================================

L = 1.0
DT_SAVE = 0.1
COMPUTE_DTYPE = np.float32
EPS0 = 1e-12


# ============================================================
# Variable-name auto detection
# ============================================================

REF_FIELD_CANDIDATES = [
    ("gt_u", "gt_v"),
    ("u_ref", "v_ref"),
]

METHOD_FIELD_CANDIDATES: Dict[str, list[Tuple[str, str]]] = {
    "FNO": [
        ("pred_u", "pred_v"),
        ("u_fno", "v_fno"),
    ],
    "GNOT": [
        ("u_gnot", "v_gnot"),
        ("pred_u_gnot", "pred_v_gnot"),
    ],
    "Ours": [
        ("u_ai", "v_ai"),
        ("u_ours", "v_ours"),
    ],
    "dns128": [
        ("u_dns128", "v_dns128"),
    ],
    "dns64": [
        ("u_dns64", "v_dns64"),
    ],
}

METHOD_ORDER = ["dns64", "dns128", "FNO", "GNOT", "Ours"]

METHOD_LABELS_TEX = {
    "Ours": r"Ours",
    "dns128": r"DNS\_128$\times$128",
    "dns64": r"DNS\_64$\times$64",
    "FNO": r"FNO",
    "GNOT": r"GNOT",
}


def pick_existing_pair(
    ds: xr.Dataset,
    candidates: list[Tuple[str, str]],
    name: str,
) -> Tuple[str, str]:
    """
    Pick the first existing (u, v) variable pair.
    """
    for u_name, v_name in candidates:
        if u_name in ds.variables and v_name in ds.variables:
            return u_name, v_name

    msg = [f"Cannot find {name} variable pair. Tried:"]
    for u_name, v_name in candidates:
        msg.append(f"  ({u_name}, {v_name})")
    msg.append("Available variables:")
    msg.extend([f"  {v}" for v in ds.variables])
    raise KeyError("\n".join(msg))


def build_method_map(ds: xr.Dataset) -> Dict[str, Tuple[str, str]]:
    """
    Build method -> (u_var, v_var) map from existing NetCDF variables.
    """
    method_map: Dict[str, Tuple[str, str]] = {}

    for method, candidates in METHOD_FIELD_CANDIDATES.items():
        for u_name, v_name in candidates:
            if u_name in ds.variables and v_name in ds.variables:
                method_map[method] = (u_name, v_name)
                break

    if not method_map:
        msg = ["No prediction method variable pairs found. Tried:"]
        for method, candidates in METHOD_FIELD_CANDIDATES.items():
            msg.append(f"  {method}: {candidates}")
        msg.append("Available variables:")
        msg.extend([f"  {v}" for v in ds.variables])
        raise KeyError("\n".join(msg))

    return method_map


# ============================================================
# CC utilities
# ============================================================

def transpose_cc_to_standard(da: xr.DataArray) -> xr.DataArray:
    """
    Normalize CC field dimensions to:
        (trajectory/case, step/time, x, y)

    Accepted dimension names:
        trajectory dimension: "trajectory", "traj", "case"
        time dimension:       "step", "time", "t"
        x dimension:          "x", "i", "nx"
        y dimension:          "y", "j", "ny"

    If dimension names cannot be inferred but da.ndim == 4,
    the current order is assumed to already be:
        (trajectory, step, x, y)
    """
    dims = list(da.dims)

    if da.ndim != 4:
        raise ValueError(
            f"Expected a 4D CC array (num_traj, num_steps, Nx, Ny), "
            f"but variable {da.name!r} has dims={da.dims}, shape={da.shape}"
        )

    traj_dim = next((d for d in dims if d in ("trajectory", "traj", "case")), None)
    step_dim = next((d for d in dims if d in ("step", "time", "t")), None)
    x_dim = next((d for d in dims if d in ("x", "i", "nx")), None)
    y_dim = next((d for d in dims if d in ("y", "j", "ny")), None)

    if traj_dim and step_dim and x_dim and y_dim:
        return da.transpose(traj_dim, step_dim, x_dim, y_dim)

    return da


def load_cc_array(ds: xr.Dataset, var_name: str) -> np.ndarray:
    """
    Load one CC variable as:
        (num_traj, num_steps, Nx, Ny)
    """
    da = transpose_cc_to_standard(ds[var_name])
    arr = da.values.astype(COMPUTE_DTYPE, copy=False)

    if arr.ndim != 4:
        raise ValueError(
            f"Variable {var_name!r} must be 4D after transpose, "
            f"but got shape={arr.shape}"
        )

    return arr


def validate_cc_pair(u: np.ndarray, v: np.ndarray, name: str) -> None:
    """
    Validate CC-grid pair:
        u: (num_traj, num_steps, Nx, Ny)
        v: (num_traj, num_steps, Nx, Ny)
    """
    if u.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"[{name}] expected 4D arrays, got u.shape={u.shape}, v.shape={v.shape}"
        )

    if u.shape != v.shape:
        raise ValueError(
            f"[{name}] u and v shapes must match for CC grid:\n"
            f"  u.shape={u.shape}\n"
            f"  v.shape={v.shape}"
        )


def infer_grid_spacing_from_cc(
    u_cc_all: np.ndarray,
    v_cc_all: np.ndarray,
) -> Tuple[int, int, float, float]:
    """
    Infer Nx, Ny, dx, dy from CC arrays.
    """
    validate_cc_pair(u_cc_all, v_cc_all, "infer_grid_spacing_from_cc")

    nx = u_cc_all.shape[-2]
    ny = u_cc_all.shape[-1]

    dx = L / nx
    dy = L / ny

    return nx, ny, dx, dy


# ============================================================
# Vorticity and corner-mask definitions
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


def corner_index_slices(
    nx: int,
    ny: int,
    corner: str,
) -> Tuple[slice, slice]:
    """
    Paper-consistent fixed bottom-corner masks on a cell-centered grid.

    n_c = floor(Nx / 3)

    I_BL = {(i,j): 2 <= i < n_c,        2 <= j < n_c}
    I_BR = {(i,j): Nx-n_c <= i < Nx-2, 2 <= j < n_c}

    Python slices are end-exclusive.

    For Nx=Ny=64:
        n_c = 21
        BL: i = 2..20,  j = 2..20
        BR: i = 43..61, j = 2..20
    """
    if nx != ny:
        raise ValueError(
            f"Expected square CC grid for cavity diagnostics, got nx={nx}, ny={ny}"
        )

    nc = nx // 3

    if nc <= 2:
        raise ValueError(f"Grid too small for corner mask: nx={nx}, nc={nc}")

    if corner == "BL":
        i_slice = slice(2, nc)
        j_slice = slice(2, nc)
    elif corner == "BR":
        i_slice = slice(nx - nc, nx - 2)
        j_slice = slice(2, nc)
    else:
        raise ValueError(f"Unsupported corner: {corner}")

    i_len = i_slice.stop - i_slice.start
    j_len = j_slice.stop - j_slice.start
    if i_len <= 0 or j_len <= 0:
        raise ValueError(
            f"Empty {corner} corner mask: "
            f"i_slice={i_slice}, j_slice={j_slice}, nx={nx}, ny={ny}, nc={nc}"
        )

    return i_slice, j_slice


def corner_vorticity_l2_errors_from_omega(
    omega_pred: np.ndarray,
    omega_ref: np.ndarray,
    corner: str,
    dx: float,
    dy: float,
    eps0: float = EPS0,
) -> Tuple[float, float, float]:
    """
    Compute paper-consistent corner-vorticity L2 errors.

    Returns:
        abs_l2:
            sqrt(sum_{I_s} (omega_pred - omega_ref)^2 dx dy)

        rel_l2:
            abs_l2 / max(sqrt(sum_{I_s} omega_ref^2 dx dy), eps0)

        ref_l2:
            sqrt(sum_{I_s} omega_ref^2 dx dy)
    """
    if omega_pred.shape != omega_ref.shape:
        raise ValueError(
            f"omega_pred and omega_ref shapes do not match: "
            f"{omega_pred.shape} vs {omega_ref.shape}"
        )

    nx, ny = omega_ref.shape
    i_slice, j_slice = corner_index_slices(nx, ny, corner)

    w_pred = omega_pred[i_slice, j_slice]
    w_ref = omega_ref[i_slice, j_slice]

    diff = w_pred - w_ref

    abs_l2 = float(np.sqrt(np.sum(diff * diff) * dx * dy))
    ref_l2 = float(np.sqrt(np.sum(w_ref * w_ref) * dx * dy))
    rel_l2 = float(abs_l2 / max(ref_l2, eps0))

    return abs_l2, rel_l2, ref_l2


# ============================================================
# Aggregation helpers
# ============================================================

def summarize_loss_by_time(
    losses: np.ndarray,
    time_rel: np.ndarray,
    quantity: str,
    method: str,
    corner: str,
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

    for k, t in enumerate(time_rel):
        yield {
            "time_rel": float(t),
            "quantity": quantity,
            "method": method,
            "corner": corner,
            "loss_mean": float(mean_t[k]),
            "loss_std": float(std_t[k]),
            "loss_median": float(median_t[k]),
            "loss_p95": float(p95_t[k]),
            "n_traj": int(n_traj),
        }


def summarize_scalar_loss(
    losses: np.ndarray,
    quantity: str,
    method: str,
    corner: str,
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
        "corner": corner,
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
    corner: str,
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
                    "corner": corner,
                    "loss": float(losses[traj_idx, step]),
                }
            )

    return rows


def summarize_ref_norm_by_time(
    ref_norms: np.ndarray,
    time_rel: np.ndarray,
    corner: str,
) -> Iterable[dict]:
    """
    ref_norms: (N_traj, T)

    Traceability output for the denominator in the relative error.
    """
    mean_t = np.nanmean(ref_norms, axis=0)
    std_t = np.nanstd(ref_norms, axis=0, ddof=0)
    n_traj = ref_norms.shape[0]

    for k, t in enumerate(time_rel):
        yield {
            "time_rel": float(t),
            "quantity": "CornerVorticityReferenceL2Norm",
            "method": "ref",
            "corner": corner,
            "value_mean": float(mean_t[k]),
            "value_std": float(std_t[k]),
            "n_traj": int(n_traj),
        }


# ============================================================
# Output writers
# ============================================================

def write_scalar_txt(df_scalar: pd.DataFrame, path: Path) -> None:
    """
    Write scalar summary to a plain text file.
    """
    lines = []
    lines.append("Paper-Consistent Corner-Vorticity Loss Summary")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  omega = dv/dx - du/dy")
    lines.append("  n_c = floor(Nx / 3)")
    lines.append("  I_BL = {(i,j): 2 <= i < n_c,        2 <= j < n_c}")
    lines.append("  I_BR = {(i,j): Nx-n_c <= i < Nx-2, 2 <= j < n_c}")
    lines.append("")
    lines.append("  e_omega_s(t) = sqrt(sum_{I_s} (omega_pred - omega_ref)^2 dx dy)")
    lines.append("  e_rel_omega_s(t) = e_omega_s(t) / max(sqrt(sum_{I_s} omega_ref^2 dx dy), EPS0)")
    lines.append("")
    lines.append("Main paper indicator:")
    lines.append("  CornerVorticityRelativeL2AvgCorners")
    lines.append("  = 0.5 * (time-mean BL relative error + time-mean BR relative error)")
    lines.append("")
    lines.append(
        f"{'Method':<12}"
        f"{'Quantity':<40}"
        f"{'Corner':<10}"
        f"{'Value':<16}"
        f"{'Std':<16}"
        f"{'Median':<16}"
        f"{'P95':<16}"
    )
    lines.append("-" * 126)

    df_print = df_scalar.copy()

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    df_print["method_rank"] = df_print["method"].map(method_rank).fillna(999)

    quantity_order = {
        "CornerVorticityAbsL2Loss": 0,
        "CornerVorticityRelativeL2Loss": 1,
        "CornerVorticityRelativeL2AvgCorners": 2,
    }
    df_print["quantity_rank"] = df_print["quantity"].map(quantity_order).fillna(999)

    corner_order = {"BL": 0, "BR": 1, "AVG_BL_BR": 2}
    df_print["corner_rank"] = df_print["corner"].map(corner_order).fillna(999)

    df_print = df_print.sort_values(["method_rank", "quantity_rank", "corner_rank"])

    for _, row in df_print.iterrows():
        lines.append(
            f"{str(row['method']):<12}"
            f"{str(row['quantity']):<40}"
            f"{str(row['corner']):<10}"
            f"{float(row['Value']):<16.6e}"
            f"{float(row['Std']):<16.6e}"
            f"{float(row['Median']):<16.6e}"
            f"{float(row['P95']):<16.6e}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(df_scalar: pd.DataFrame, path: Path) -> None:
    """
    Write a compact LaTeX table for the paper's main corner-vorticity indicator.

    Reports:
        CornerVorticityRelativeL2AvgCorners
    """
    quantity = "CornerVorticityRelativeL2AvgCorners"
    corner = "AVG_BL_BR"

    table = {}

    for method in METHOD_ORDER:
        sub = df_scalar[
            (df_scalar["method"] == method)
            & (df_scalar["quantity"] == quantity)
            & (df_scalar["corner"] == corner)
        ]

        if sub.empty:
            table[method] = (np.nan, np.nan)
        else:
            row = sub.iloc[0]
            table[method] = (float(row["Value"]), float(row["Std"]))

    vals = [v[0] for v in table.values() if not np.isnan(v[0])]
    best = min(vals) if vals else np.nan

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Averaged relative bottom-corner vorticity error.}",
        r"\label{tab:corner_vorticity_relative_l2}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Method & $e^{\mathrm{rel}}_{\omega,\mathrm{corner}}$ \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        mean_val, std_val = table.get(method, (np.nan, np.nan))
        label = METHOD_LABELS_TEX.get(method, method)

        if np.isnan(mean_val):
            text = "--"
        else:
            text = f"{mean_val:.4e} $\\pm$ {std_val:.1e}"
            if np.isclose(mean_val, best):
                text = r"\textbf{" + text + "}"

        lines.append(f"{label} & {text} " + r"\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def print_main_indicator(df_scalar: pd.DataFrame) -> None:
    """
    Print the main paper indicator in terminal.
    """
    quantity = "CornerVorticityRelativeL2AvgCorners"
    corner = "AVG_BL_BR"

    print("\nMain paper corner-vorticity indicator:")
    print("-" * 78)
    print(f"{'Method':<12}{'Value':<18}{'Std':<18}{'Median':<18}{'P95':<18}")
    print("-" * 78)

    for method in METHOD_ORDER:
        sub = df_scalar[
            (df_scalar["method"] == method)
            & (df_scalar["quantity"] == quantity)
            & (df_scalar["corner"] == corner)
        ]

        if sub.empty:
            continue

        row = sub.iloc[0]
        print(
            f"{method:<12}"
            f"{float(row['Value']):<18.6e}"
            f"{float(row['Std']):<18.6e}"
            f"{float(row['Median']):<18.6e}"
            f"{float(row['P95']):<18.6e}"
        )

    print("-" * 78)


# ============================================================
# Main pipeline
# ============================================================

def main(
    nc_file: str = NC_FILE,
    output_dir: str = OUTPUT_DIR,
    max_trajs: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nc_path = Path(nc_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.exists():
        raise FileNotFoundError(f"Input NetCDF file does not exist: {nc_path}")

    output_timeseries_csv = out_dir / "corner_vorticity_loss_timeseries.csv"
    output_per_traj_csv = out_dir / "corner_vorticity_per_traj_step_losses.csv"
    output_ref_norm_csv = out_dir / "corner_vorticity_ref_norm_timeseries.csv"
    output_scalar_csv = out_dir / "corner_vorticity_scalar_summary.csv"
    output_scalar_txt = out_dir / "corner_vorticity_scalar_summary.txt"
    output_tex = out_dir / "corner_vorticity_scalar_summary.tex"

    print("=" * 72)
    print("Paper-consistent CC-grid corner-vorticity diagnostic-loss pipeline")
    print("=" * 72)
    print(f"Input NetCDF: {nc_path}")
    print(f"Output dir:   {out_dir}")

    with xr.open_dataset(nc_path) as ds:
        ref_u_name, ref_v_name = pick_existing_pair(
            ds,
            REF_FIELD_CANDIDATES,
            name="reference",
        )

        method_map = build_method_map(ds)
        eval_methods = [m for m in METHOD_ORDER if m in method_map]

        print(f"Reference variables: {ref_u_name}, {ref_v_name}")
        print("Detected methods:")
        for method in eval_methods:
            u_name, v_name = method_map[method]
            print(f"  {method}: {u_name}, {v_name}")

        if not eval_methods:
            raise RuntimeError("No methods from METHOD_ORDER were detected.")

        # Load reference first to infer shape and spacing.
        u_ref_all = load_cc_array(ds, ref_u_name)
        v_ref_all = load_cc_array(ds, ref_v_name)
        validate_cc_pair(u_ref_all, v_ref_all, "reference")

        n_traj_total, n_steps, nx, ny = u_ref_all.shape
        n_traj = n_traj_total if max_trajs is None else min(max_trajs, n_traj_total)

        nx, ny, dx, dy = infer_grid_spacing_from_cc(u_ref_all, v_ref_all)

        # Print mask details.
        nc = nx // 3
        bl_i, bl_j = corner_index_slices(nx, ny, "BL")
        br_i, br_j = corner_index_slices(nx, ny, "BR")

        time_rel = np.arange(1, n_steps + 1, dtype=np.float64) * DT_SAVE

        print(f"Trajectories: {n_traj}/{n_traj_total}")
        print(f"Steps:        {n_steps}")
        print("CC grid:")
        print(f"  {ref_u_name}.shape = {u_ref_all.shape} = (traj, step, Nx, Ny)")
        print(f"  {ref_v_name}.shape = {v_ref_all.shape} = (traj, step, Nx, Ny)")
        print(f"  inferred CC grid = {nx} x {ny}")
        print(f"  dx = {dx:.6e}, dy = {dy:.6e}")
        print(f"Corner masks:")
        print(f"  n_c = floor(Nx/3) = {nc}")
        print(f"  BL i={bl_i.start}:{bl_i.stop}, j={bl_j.start}:{bl_j.stop}")
        print(f"  BR i={br_i.start}:{br_i.stop}, j={br_j.start}:{br_j.stop}")

        # Load all method arrays once.
        cc_cache_all = {
            "ref": (u_ref_all[:n_traj], v_ref_all[:n_traj])
        }

        for method in eval_methods:
            u_name, v_name = method_map[method]

            u_all = load_cc_array(ds, u_name)
            v_all = load_cc_array(ds, v_name)
            validate_cc_pair(u_all, v_all, method)

            if u_all.shape != u_ref_all.shape or v_all.shape != v_ref_all.shape:
                raise ValueError(
                    f"[{method}] shape mismatch with reference:\n"
                    f"  u_all.shape={u_all.shape}, v_all.shape={v_all.shape}\n"
                    f"  u_ref.shape={u_ref_all.shape}, v_ref.shape={v_ref_all.shape}"
                )

            cc_cache_all[method] = (u_all[:n_traj], v_all[:n_traj])

        abs_loss = {
            corner: {
                method: np.zeros((n_traj, n_steps), dtype=np.float64)
                for method in eval_methods
            }
            for corner in ("BL", "BR")
        }

        rel_loss = {
            corner: {
                method: np.zeros((n_traj, n_steps), dtype=np.float64)
                for method in eval_methods
            }
            for corner in ("BL", "BR")
        }

        ref_norm = {
            corner: np.zeros((n_traj, n_steps), dtype=np.float64)
            for corner in ("BL", "BR")
        }

        # ------------------------------------------------------------
        # Compute per-trajectory, per-time corner vorticity losses
        # ------------------------------------------------------------
        u_ref_use, v_ref_use = cc_cache_all["ref"]

        for traj_idx in range(n_traj):
            for step in range(n_steps):
                omega_ref = compute_vorticity_cc(
                    u_ref_use[traj_idx, step],
                    v_ref_use[traj_idx, step],
                    dx,
                    dy,
                )

                for method in eval_methods:
                    u_pred_all, v_pred_all = cc_cache_all[method]

                    omega_pred = compute_vorticity_cc(
                        u_pred_all[traj_idx, step],
                        v_pred_all[traj_idx, step],
                        dx,
                        dy,
                    )

                    for corner in ("BL", "BR"):
                        abs_l2, rel_l2, ref_l2 = corner_vorticity_l2_errors_from_omega(
                            omega_pred=omega_pred,
                            omega_ref=omega_ref,
                            corner=corner,
                            dx=dx,
                            dy=dy,
                            eps0=EPS0,
                        )

                        abs_loss[corner][method][traj_idx, step] = abs_l2
                        rel_loss[corner][method][traj_idx, step] = rel_l2
                        ref_norm[corner][traj_idx, step] = ref_l2

            if (traj_idx + 1) % 5 == 0 or traj_idx == n_traj - 1:
                print(f"Finished {traj_idx + 1}/{n_traj} trajectories")

    # ------------------------------------------------------------
    # Build outputs
    # ------------------------------------------------------------
    timeseries_rows = []
    per_traj_rows = []
    scalar_rows = []
    ref_norm_rows = []

    for corner in ("BL", "BR"):
        ref_norm_rows.extend(
            summarize_ref_norm_by_time(
                ref_norms=ref_norm[corner],
                time_rel=time_rel,
                corner=corner,
            )
        )

    for method in eval_methods:
        for corner in ("BL", "BR"):
            timeseries_rows.extend(
                summarize_loss_by_time(
                    losses=abs_loss[corner][method],
                    time_rel=time_rel,
                    quantity="CornerVorticityAbsL2Loss",
                    method=method,
                    corner=corner,
                )
            )

            timeseries_rows.extend(
                summarize_loss_by_time(
                    losses=rel_loss[corner][method],
                    time_rel=time_rel,
                    quantity="CornerVorticityRelativeL2Loss",
                    method=method,
                    corner=corner,
                )
            )

            scalar_rows.append(
                summarize_scalar_loss(
                    losses=abs_loss[corner][method],
                    quantity="CornerVorticityAbsL2Loss",
                    method=method,
                    corner=corner,
                )
            )

            scalar_rows.append(
                summarize_scalar_loss(
                    losses=rel_loss[corner][method],
                    quantity="CornerVorticityRelativeL2Loss",
                    method=method,
                    corner=corner,
                )
            )

            per_traj_rows.extend(
                build_per_traj_step_loss_rows(
                    losses=abs_loss[corner][method],
                    time_rel=time_rel,
                    quantity="CornerVorticityAbsL2Loss",
                    method=method,
                    corner=corner,
                )
            )

            per_traj_rows.extend(
                build_per_traj_step_loss_rows(
                    losses=rel_loss[corner][method],
                    time_rel=time_rel,
                    quantity="CornerVorticityRelativeL2Loss",
                    method=method,
                    corner=corner,
                )
            )

        rel_avg_corners = 0.5 * (
            rel_loss["BL"][method] + rel_loss["BR"][method]
        )

        timeseries_rows.extend(
            summarize_loss_by_time(
                losses=rel_avg_corners,
                time_rel=time_rel,
                quantity="CornerVorticityRelativeL2AvgCorners",
                method=method,
                corner="AVG_BL_BR",
            )
        )

        scalar_rows.append(
            summarize_scalar_loss(
                losses=rel_avg_corners,
                quantity="CornerVorticityRelativeL2AvgCorners",
                method=method,
                corner="AVG_BL_BR",
            )
        )

        per_traj_rows.extend(
            build_per_traj_step_loss_rows(
                losses=rel_avg_corners,
                time_rel=time_rel,
                quantity="CornerVorticityRelativeL2AvgCorners",
                method=method,
                corner="AVG_BL_BR",
            )
        )

    df_timeseries = pd.DataFrame(timeseries_rows)
    df_per_traj = pd.DataFrame(per_traj_rows)
    df_ref_norm = pd.DataFrame(ref_norm_rows)
    df_scalar = pd.DataFrame(scalar_rows)

    df_timeseries.to_csv(output_timeseries_csv, index=False)
    df_per_traj.to_csv(output_per_traj_csv, index=False)
    df_ref_norm.to_csv(output_ref_norm_csv, index=False)
    df_scalar.to_csv(output_scalar_csv, index=False)

    write_scalar_txt(df_scalar, output_scalar_txt)
    write_latex_table(df_scalar, output_tex)
    print_main_indicator(df_scalar)

    print("\nSaved files:")
    print(f"  {output_timeseries_csv}")
    print(f"  {output_per_traj_csv}")
    print(f"  {output_ref_norm_csv}")
    print(f"  {output_scalar_csv}")
    print(f"  {output_scalar_txt}")
    print(f"  {output_tex}")

    print("\nDone.")

    return df_timeseries, df_scalar


if __name__ == "__main__":
    main()