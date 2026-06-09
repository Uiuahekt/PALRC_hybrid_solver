"""
Compute absolute centerline ensemble-RMS L2 losses from a MAC-grid NetCDF file.

Input NetCDF is assumed to store MAC/staggered velocity fields:
    u: (num_traj, num_steps, Nx+1, Ny)
    v: (num_traj, num_steps, Nx,   Ny+1)

Procedure:
  1. Read MAC velocity fields.
  2. Convert MAC fields to cell-centered fields:
         u_cc = 0.5 * (u_mac[:-1, :] + u_mac[1:, :])
         v_cc = 0.5 * (v_mac[:, :-1] + v_mac[:, 1:])
  3. Extract centerline profiles:
         u(0.5, y, t)
         v(x, 0.5, t)
  4. Compute ensemble-RMS residual heatmaps:
         Hu_m(y,t) = sqrt(mean_i((u_pred_i(0.5,y,t)-u_ref_i(0.5,y,t))^2))
         Hv_m(x,t) = sqrt(mean_i((v_pred_i(x,0.5,t)-v_ref_i(x,0.5,t))^2))
  5. Compute scalar losses:
         E_u_cl = mean_t sqrt(mean_y Hu_m(y,t)^2)
         E_v_cl = mean_t sqrt(mean_x Hv_m(x,t)^2)
         E_cl   = 0.5 * (E_u_cl + E_v_cl)

This is an absolute centerline ensemble-RMS L2 loss averaged over time.
It is not a relative L2 loss.

Outputs:
    centerline_rms_heatmaps.npz
    centerline_abs_ensemble_rms_l2_loss_table.csv
    centerline_abs_ensemble_rms_l2_loss_table.tex
    centerline_abs_ensemble_rms_l2_timeseries.csv
    centerline_scalar_Ecl.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# User settings: only edit these two paths
# ============================================================

NC_FILE = "../data/full_trajectories_MAC_32bit.nc"
OUTPUT_DIR = "../results/center_line_error"


# ============================================================
# Configuration
# ============================================================

COMPUTE_DTYPE = np.float32

L = 1.0
DT_SAVE = 0.1
N_PROFILE = 65

REF_U_KEY = "u_ref"
REF_V_KEY = "v_ref"

METHODS = {
    "AI_FNO": ("u_ai", "v_ai"),
    "DNS_128": ("u_dns128", "v_dns128"),
    "DNS_64": ("u_dns64", "v_dns64"),
}

METHOD_ORDER = ["AI_FNO", "DNS_128", "DNS_64"]

METHOD_LABELS_TEX = {
    "AI_FNO": r"AI~(FNO)",
    "DNS_128": r"DNS\_128$\times$128",
    "DNS_64": r"DNS\_64$\times$64",
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

    if u.ndim < 4:
        raise ValueError(
            f"[{name}] expected at least 4D arrays "
            f"(num_traj, num_steps, ..., ...), "
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


def mac_to_cell_center(
    u_mac: np.ndarray,
    v_mac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one MAC-grid velocity field to cell-centered velocity field.

    u_mac: (Nx+1, Ny)
    v_mac: (Nx, Ny+1)

    Returns:
        u_cc: (Nx, Ny)
        v_cc: (Nx, Ny)
    """
    half = COMPUTE_DTYPE(0.5)
    u_cc = half * (u_mac[:-1, :] + u_mac[1:, :])
    v_cc = half * (v_mac[:, :-1] + v_mac[:, 1:])

    return (
        u_cc.astype(COMPUTE_DTYPE, copy=False),
        v_cc.astype(COMPUTE_DTYPE, copy=False),
    )


def cc_axes(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Cell-centered coordinates on [0, L] x [0, L].
    """
    dx = L / nx
    dy = L / ny

    x = np.linspace(dx / 2.0, L - dx / 2.0, nx, dtype=np.float64)
    y = np.linspace(dy / 2.0, L - dy / 2.0, ny, dtype=np.float64)

    return x, y, dx, dy


def centerline_u_of_y(
    u_cc: np.ndarray,
    y_query: np.ndarray,
    x_center: float = 0.5 * L,
) -> np.ndarray:
    """
    Extract u(0.5, y) from cell-centered u field by linear interpolation.
    """
    nx, ny = u_cc.shape
    x_cc, y_cc, _, _ = cc_axes(nx, ny)

    ix = np.clip(np.searchsorted(x_cc, x_center) - 1, 0, nx - 2)
    fx = (x_center - x_cc[ix]) / (x_cc[ix + 1] - x_cc[ix])

    u_line_on_y_cc = (1.0 - fx) * u_cc[ix, :] + fx * u_cc[ix + 1, :]
    u_line = np.interp(y_query, y_cc, u_line_on_y_cc)

    return u_line.astype(np.float64)


def centerline_v_of_x(
    v_cc: np.ndarray,
    x_query: np.ndarray,
    y_center: float = 0.5 * L,
) -> np.ndarray:
    """
    Extract v(x, 0.5) from cell-centered v field by linear interpolation.
    """
    nx, ny = v_cc.shape
    x_cc, y_cc, _, _ = cc_axes(nx, ny)

    iy = np.clip(np.searchsorted(y_cc, y_center) - 1, 0, ny - 2)
    fy = (y_center - y_cc[iy]) / (y_cc[iy + 1] - y_cc[iy])

    v_line_on_x_cc = (1.0 - fy) * v_cc[:, iy] + fy * v_cc[:, iy + 1]
    v_line = np.interp(x_query, x_cc, v_line_on_x_cc)

    return v_line.astype(np.float64)


# ============================================================
# Centerline RMS and scalar losses
# ============================================================

def compute_centerline_rms_heatmaps_for_method(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble-RMS centerline residual heatmaps.

    Inputs:
        u_pred/u_ref: (num_traj, num_steps, Nx+1, Ny)
        v_pred/v_ref: (num_traj, num_steps, Nx, Ny+1)

    Returns:
        Hu: (n_y_query, num_steps)
        Hv: (n_x_query, num_steps)
    """
    validate_mac_pair(u_pred, v_pred, "prediction")
    validate_mac_pair(u_ref, v_ref, "reference")

    if u_pred.shape != u_ref.shape:
        raise ValueError(
            f"u_pred and u_ref shapes do not match:\n"
            f"  u_pred.shape={u_pred.shape}\n"
            f"  u_ref.shape={u_ref.shape}"
        )

    if v_pred.shape != v_ref.shape:
        raise ValueError(
            f"v_pred and v_ref shapes do not match:\n"
            f"  v_pred.shape={v_pred.shape}\n"
            f"  v_ref.shape={v_ref.shape}"
        )

    num_traj, num_steps = u_ref.shape[:2]

    hu_sq_sum = np.zeros((len(y_query), num_steps), dtype=np.float64)
    hv_sq_sum = np.zeros((len(x_query), num_steps), dtype=np.float64)

    for i in range(num_traj):
        for step in range(num_steps):
            u_p_cc, v_p_cc = mac_to_cell_center(u_pred[i, step], v_pred[i, step])
            u_r_cc, v_r_cc = mac_to_cell_center(u_ref[i, step], v_ref[i, step])

            u_p_line = centerline_u_of_y(u_p_cc, y_query)
            u_r_line = centerline_u_of_y(u_r_cc, y_query)

            v_p_line = centerline_v_of_x(v_p_cc, x_query)
            v_r_line = centerline_v_of_x(v_r_cc, x_query)

            du_line = u_p_line - u_r_line
            dv_line = v_p_line - v_r_line

            hu_sq_sum[:, step] += du_line * du_line
            hv_sq_sum[:, step] += dv_line * dv_line

    Hu = np.sqrt(hu_sq_sum / float(num_traj))
    Hv = np.sqrt(hv_sq_sum / float(num_traj))

    return Hu, Hv


def compute_centerline_scalar_metrics(
    Hu: np.ndarray,
    Hv: np.ndarray,
    time_rel: np.ndarray,
) -> dict[str, float]:
    """
    Compute scalar centerline losses from ensemble-RMS residual heatmaps.

    Hu shape: (n_y, n_time)
    Hv shape: (n_x, n_time)

    Main reported metrics:
        E_u_cl = mean_t sqrt(mean_y Hu(y,t)^2)
        E_v_cl = mean_t sqrt(mean_x Hv(x,t)^2)
        E_cl   = 0.5 * (E_u_cl + E_v_cl)
    """
    u_l2_t = np.sqrt(np.nanmean(Hu * Hu, axis=0))
    v_l2_t = np.sqrt(np.nanmean(Hv * Hv, axis=0))

    e_u = float(np.nanmean(u_l2_t))
    e_v = float(np.nanmean(v_l2_t))

    u_mean_st = float(np.nanmean(Hu))
    v_mean_st = float(np.nanmean(Hv))

    return {
        "time_start": float(np.nanmin(time_rel)),
        "time_end": float(np.nanmax(time_rel)),
        "n_time": int(len(time_rel)),
        "n_coord_u": int(Hu.shape[0]),
        "n_coord_v": int(Hv.shape[0]),
        "E_u_cl_abs_ensemble_rms_L2_time_mean": e_u,
        "E_v_cl_abs_ensemble_rms_L2_time_mean": e_v,
        "E_cl_abs_ensemble_rms_L2_time_mean": 0.5 * (e_u + e_v),
        "u_centerline_abs_ensemble_rms_spacetime_mean": u_mean_st,
        "v_centerline_abs_ensemble_rms_spacetime_mean": v_mean_st,
        "centerline_abs_ensemble_rms_spacetime_mean": 0.5 * (u_mean_st + v_mean_st),
    }


def build_centerline_timeseries_dataframe(
    method: str,
    Hu: np.ndarray,
    Hv: np.ndarray,
    time_rel: np.ndarray,
) -> pd.DataFrame:
    """
    Build time series:
        E_u_cl(t) = sqrt(mean_y Hu(y,t)^2)
        E_v_cl(t) = sqrt(mean_x Hv(x,t)^2)
        E_cl(t)   = 0.5 * (E_u_cl(t) + E_v_cl(t))
    """
    u_l2_t = np.sqrt(np.nanmean(Hu * Hu, axis=0))
    v_l2_t = np.sqrt(np.nanmean(Hv * Hv, axis=0))
    e_cl_t = 0.5 * (u_l2_t + v_l2_t)

    rows = []
    for k, t in enumerate(time_rel):
        rows.append(
            {
                "Method": method,
                "time_rel": float(t),
                "E_u_cl_abs_ensemble_rms_L2": float(u_l2_t[k]),
                "E_v_cl_abs_ensemble_rms_L2": float(v_l2_t[k]),
                "E_cl_abs_ensemble_rms_L2": float(e_cl_t[k]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Output writers
# ============================================================

def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    """
    Write LaTeX table for scalar centerline losses.
    """
    best = {
        "u": df["E_u_cl_abs_ensemble_rms_L2_time_mean"].min(),
        "v": df["E_v_cl_abs_ensemble_rms_L2_time_mean"].min(),
        "avg": df["E_cl_abs_ensemble_rms_L2_time_mean"].min(),
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Absolute centerline ensemble-RMS $L^2$ losses averaged over time.}",
        r"\label{tab:centerline_abs_ensemble_rms_l2_loss}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & $\mathcal{E}_{u,\mathrm{cl}}$ & $\mathcal{E}_{v,\mathrm{cl}}$ & $\mathcal{E}_{\mathrm{cl}}$ \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        method = row["Method"]
        label = METHOD_LABELS_TEX.get(method, method)

        vals = [
            ("u", float(row["E_u_cl_abs_ensemble_rms_L2_time_mean"])),
            ("v", float(row["E_v_cl_abs_ensemble_rms_L2_time_mean"])),
            ("avg", float(row["E_cl_abs_ensemble_rms_L2_time_mean"])),
        ]

        cells = []
        for key, value in vals:
            text = f"{value:.4e}"
            if np.isclose(value, best[key]):
                text = r"\textbf{" + text + "}"
            cells.append(text)

        lines.append(
            f"{label} & {cells[0]} & {cells[1]} & {cells[2]} " + r"\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_scalar_txt(df: pd.DataFrame, path: Path) -> None:
    """
    Write scalar centerline losses to a plain text file.
    """
    lines = []
    lines.append("Absolute Centerline Ensemble-RMS L2 Losses")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  Hu(y,t) = sqrt(mean_i((u_pred_i(0.5,y,t) - u_ref_i(0.5,y,t))^2))")
    lines.append("  Hv(x,t) = sqrt(mean_i((v_pred_i(x,0.5,t) - v_ref_i(x,0.5,t))^2))")
    lines.append("")
    lines.append("  E_u_cl = mean_t sqrt(mean_y Hu(y,t)^2)")
    lines.append("  E_v_cl = mean_t sqrt(mean_x Hv(x,t)^2)")
    lines.append("  E_cl   = 0.5 * (E_u_cl + E_v_cl)")
    lines.append("")
    lines.append("Note:")
    lines.append("  These are absolute centerline ensemble-RMS L2 losses.")
    lines.append("  They are not relative L2 losses.")
    lines.append("")
    lines.append(
        f"{'Method':<16}"
        f"{'E_u_cl':<18}"
        f"{'E_v_cl':<18}"
        f"{'E_cl':<18}"
    )
    lines.append("-" * 70)

    for _, row in df.iterrows():
        method = row["Method"]
        e_u = row["E_u_cl_abs_ensemble_rms_L2_time_mean"]
        e_v = row["E_v_cl_abs_ensemble_rms_L2_time_mean"]
        e_cl = row["E_cl_abs_ensemble_rms_L2_time_mean"]

        lines.append(
            f"{method:<16}"
            f"{e_u:<18.6e}"
            f"{e_v:<18.6e}"
            f"{e_cl:<18.6e}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_scalar_console_table(df: pd.DataFrame) -> None:
    """
    Print scalar E_u_cl, E_v_cl, E_cl clearly in terminal.
    """
    print("\nScalar centerline losses:")
    print("-" * 70)
    print(
        f"{'Method':<16}"
        f"{'E_u_cl':<18}"
        f"{'E_v_cl':<18}"
        f"{'E_cl':<18}"
    )
    print("-" * 70)

    for _, row in df.iterrows():
        print(
            f"{row['Method']:<16}"
            f"{row['E_u_cl_abs_ensemble_rms_L2_time_mean']:<18.6e}"
            f"{row['E_v_cl_abs_ensemble_rms_L2_time_mean']:<18.6e}"
            f"{row['E_cl_abs_ensemble_rms_L2_time_mean']:<18.6e}"
        )

    print("-" * 70)


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    nc_path = Path(NC_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.exists():
        raise FileNotFoundError(f"Input NetCDF file does not exist: {nc_path}")

    output_npz = output_dir / "centerline_rms_heatmaps.npz"
    output_csv = output_dir / "centerline_abs_ensemble_rms_l2_loss_table.csv"
    output_tex = output_dir / "centerline_abs_ensemble_rms_l2_loss_table.tex"
    output_ts_csv = output_dir / "centerline_abs_ensemble_rms_l2_timeseries.csv"
    output_scalar_txt = output_dir / "centerline_scalar_Ecl.txt"

    print("Reading MAC-grid NetCDF file:")
    print(f"  {nc_path}")
    print("Saving outputs to:")
    print(f"  {output_dir}")

    ds = xr.open_dataset(nc_path)

    required_vars = [REF_U_KEY, REF_V_KEY]
    for u_key, v_key in METHODS.values():
        required_vars.extend([u_key, v_key])

    missing = [var for var in required_vars if var not in ds.variables]
    if missing:
        ds.close()
        raise KeyError(
            "Missing variables in NetCDF file:\n  "
            + "\n  ".join(missing)
        )

    u_ref = ds[REF_U_KEY].values.astype(COMPUTE_DTYPE, copy=False)
    v_ref = ds[REF_V_KEY].values.astype(COMPUTE_DTYPE, copy=False)

    validate_mac_pair(u_ref, v_ref, "reference")

    num_traj, num_steps = u_ref.shape[:2]
    nx = v_ref.shape[-2]
    ny = u_ref.shape[-1]

    x_query = np.linspace(0.0, L, N_PROFILE, dtype=np.float64)
    y_query = np.linspace(0.0, L, N_PROFILE, dtype=np.float64)

    time_rel = np.arange(1, num_steps + 1, dtype=np.float64) * DT_SAVE

    print(f"Trajectories: {num_traj}")
    print(f"Steps:        {num_steps}")
    print("MAC grid:")
    print(f"  u_ref shape = {u_ref.shape} = (traj, step, Nx+1, Ny)")
    print(f"  v_ref shape = {v_ref.shape} = (traj, step, Nx, Ny+1)")
    print(f"  inferred cell-centered grid = {nx} x {ny}")
    print(f"Centerline profile points: {N_PROFILE}")

    rows = []
    ts_dfs = []

    npz_dict = {
        "time_rel": time_rel,
        "x_query": x_query,
        "y_query": y_query,
    }

    for method in METHOD_ORDER:
        if method not in METHODS:
            continue

        u_key, v_key = METHODS[method]

        print(f"\nComputing centerline ensemble-RMS loss for method: {method}")

        u_pred = ds[u_key].values.astype(COMPUTE_DTYPE, copy=False)
        v_pred = ds[v_key].values.astype(COMPUTE_DTYPE, copy=False)

        validate_mac_pair(u_pred, v_pred, method)

        Hu, Hv = compute_centerline_rms_heatmaps_for_method(
            u_pred=u_pred,
            v_pred=v_pred,
            u_ref=u_ref,
            v_ref=v_ref,
            x_query=x_query,
            y_query=y_query,
        )

        row = {"Method": method}
        row.update(compute_centerline_scalar_metrics(Hu, Hv, time_rel))
        rows.append(row)

        ts_dfs.append(
            build_centerline_timeseries_dataframe(method, Hu, Hv, time_rel)
        )

        npz_dict[f"Hu_{method}"] = Hu
        npz_dict[f"Hv_{method}"] = Hv

        print(
            f"  E_u_cl = {row['E_u_cl_abs_ensemble_rms_L2_time_mean']:.6e}, "
            f"E_v_cl = {row['E_v_cl_abs_ensemble_rms_L2_time_mean']:.6e}, "
            f"E_cl = {row['E_cl_abs_ensemble_rms_L2_time_mean']:.6e}"
        )

    ds.close()

    df = pd.DataFrame(rows)
    df_ts = pd.concat(ts_dfs, ignore_index=True)

    df.to_csv(output_csv, index=False)
    df_ts.to_csv(output_ts_csv, index=False)
    np.savez_compressed(output_npz, **npz_dict)
    write_latex_table(df, output_tex)
    write_scalar_txt(df, output_scalar_txt)

    write_scalar_console_table(df)

    print("\nSaved files:")
    print(f"  {output_npz}")
    print(f"  {output_csv}")
    print(f"  {output_tex}")
    print(f"  {output_ts_csv}")
    print(f"  {output_scalar_txt}")


if __name__ == "__main__":
    main()