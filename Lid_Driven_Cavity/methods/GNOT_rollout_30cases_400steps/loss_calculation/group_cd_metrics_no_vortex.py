"""
Group C/D kinetic-energy and enstrophy loss export from a CC-grid NetCDF file.

This script computes only the following two diagnostics:
  1. Kinetic energy
  2. Total enstrophy

Secondary-vortex / corner-vorticity diagnostics are not computed here.

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

Aggregation order:
  1. For each held-out trajectory and each time step, compute the diagnostic
     for the prediction and for its own reference trajectory.
  2. Convert that pair into a per-trajectory loss.
  3. Aggregate the loss over trajectories at each time step.
  4. For scalar summaries, first average each trajectory over time, then
     aggregate over trajectories.

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

NC_FILE = "../data/gnot_predictions_30cases_400steps.nc"
OUTPUT_DIR = "../results/group_cd_metrics_no_vortex"


# ============================================================
# Configuration
# ============================================================

L = 1.0
DT_SAVE = 0.1
COMPUTE_DTYPE = np.float32
EPS = 1e-12


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
    print("CC-grid kinetic-energy / enstrophy diagnostic-loss pipeline")
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

        if not eval_methods:
            raise RuntimeError("No methods from METHOD_ORDER were detected.")

        print(f"Reference variables: {ref_u_name}, {ref_v_name}")
        print("Detected methods:")
        for method in eval_methods:
            u_name, v_name = method_map[method]
            print(f"  {method}: {u_name}, {v_name}")

        u_ref_all = load_cc_array(ds, ref_u_name)
        v_ref_all = load_cc_array(ds, ref_v_name)
        validate_cc_pair(u_ref_all, v_ref_all, "reference")

        n_traj_total, n_steps, nx, ny = u_ref_all.shape
        n_traj = n_traj_total if max_trajs is None else min(max_trajs, n_traj_total)

        nx, ny, dx, dy = infer_grid_spacing_from_cc(u_ref_all, v_ref_all)

        time_rel = np.arange(1, n_steps + 1, dtype=np.float64) * DT_SAVE

        print(f"Trajectories: {n_traj}/{n_traj_total}")
        print(f"Steps:        {n_steps}")
        print("CC grid:")
        print(f"  {ref_u_name}.shape = {u_ref_all.shape} = (traj, step, Nx, Ny)")
        print(f"  {ref_v_name}.shape = {v_ref_all.shape} = (traj, step, Nx, Ny)")
        print(f"  inferred CC grid = {nx} x {ny}")
        print(f"  dx = {dx:.6e}, dy = {dy:.6e}")

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

        methods_all = ["ref"] + eval_methods

        ke = {
            method: np.zeros((n_traj, n_steps), dtype=np.float64)
            for method in methods_all
        }

        enstrophy = {
            method: np.zeros((n_traj, n_steps), dtype=np.float64)
            for method in methods_all
        }

        # ------------------------------------------------------------
        # Compute raw diagnostics per trajectory and time step
        # ------------------------------------------------------------
        for traj_idx in range(n_traj):
            for step in range(n_steps):
                for method in methods_all:
                    u_all, v_all = cc_cache_all[method]

                    u_cc = u_all[traj_idx, step]
                    v_cc = v_all[traj_idx, step]

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
    for method in methods_all:
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
    for method in eval_methods:
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