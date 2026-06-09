"""
从 CC / cell-centered 格式 NetCDF 文件中读取轨迹，
计算各方法相对 L2 误差与 L_infinity 误差，
按步骤对所有 seed 求均值与标准差，输出为 txt。

Input NetCDF is assumed to store cell-centered / collocated velocity fields:
    u: (num_traj, num_steps, Nx, Ny)
    v: (num_traj, num_steps, Nx, Ny)

Typical FNO-style variables:
    gt_u, gt_v
    pred_u, pred_v

Typical Ours/DNS-style variables:
    u_ref, v_ref
    u_ai, v_ai
    u_dns128, v_dns128
    u_dns64, v_dns64

Metrics:
    RelL2(t) =
        sqrt(sum((u_pred-u_ref)^2) + sum((v_pred-v_ref)^2))
        /
        sqrt(sum(u_ref^2) + sum(v_ref^2))

    Linf(t) =
        max_{x,y} sqrt((u_pred-u_ref)^2 + (v_pred-v_ref)^2)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr


COMPUTE_DTYPE = np.float32


# ============================================================
# User settings: only edit these two paths
# ============================================================

NC_FILE = "../data/fno_predictions_30cases_20steps.nc"
OUTPUT_DIR = "../results/cell_center_velocity_error"


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
    "AI  (FNO)": [
        ("u_ai", "v_ai"),
    ],
    "Ours": [
        ("u_ours", "v_ours"),
        ("u_ai", "v_ai"),
    ],
    "DNS-128": [
        ("u_dns128", "v_dns128"),
    ],
    "DNS-64": [
        ("u_dns64", "v_dns64"),
    ],
}

METHOD_ORDER = ["DNS-64", "DNS-128", "FNO", "GNOT", "AI  (FNO)", "Ours"]


# ============================================================
# Dataset utilities
# ============================================================

def pick_existing_pair(
    ds: xr.Dataset,
    candidates: list[tuple[str, str]],
    name: str,
) -> tuple[str, str]:
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
    msg.extend([f"  {var}" for var in ds.variables])
    raise KeyError("\n".join(msg))


def build_method_map(ds: xr.Dataset) -> dict[str, tuple[str, str]]:
    """
    Build method -> (u_var, v_var) map from existing NetCDF variables.
    """
    method_map: dict[str, tuple[str, str]] = {}

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
        msg.extend([f"  {var}" for var in ds.variables])
        raise KeyError("\n".join(msg))

    return method_map


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


# ============================================================
# Error computation
# ============================================================

def compute_error_cell_center(
    u_pred_cc: np.ndarray,
    v_pred_cc: np.ndarray,
    u_ref_cc: np.ndarray,
    v_ref_cc: np.ndarray,
) -> tuple[float, float]:
    """
    在 CC / 体心网格上计算误差。

    Returns:
        rel_l2:
            sqrt(sum(du^2)+sum(dv^2)) / sqrt(sum(u_ref^2)+sum(v_ref^2))

        max_err:
            max sqrt(du^2 + dv^2)
    """
    du = (u_pred_cc - u_ref_cc).astype(COMPUTE_DTYPE, copy=False)
    dv = (v_pred_cc - v_ref_cc).astype(COMPUTE_DTYPE, copy=False)

    sq_diff = (
        np.sum(du * du, dtype=COMPUTE_DTYPE)
        + np.sum(dv * dv, dtype=COMPUTE_DTYPE)
    )

    sq_ref = (
        np.sum(u_ref_cc * u_ref_cc, dtype=COMPUTE_DTYPE)
        + np.sum(v_ref_cc * v_ref_cc, dtype=COMPUTE_DTYPE)
        + COMPUTE_DTYPE(1e-12)
    )

    rel_l2 = np.sqrt(sq_diff / sq_ref).astype(COMPUTE_DTYPE)

    error_magnitude = np.sqrt(du * du + dv * dv).astype(COMPUTE_DTYPE)
    max_err = np.max(error_magnitude).astype(COMPUTE_DTYPE)

    return float(rel_l2), float(max_err)


def compute_errors_for_method(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对形状 (num_traj, num_steps, Nx, Ny) 的 CC 场批量计算误差。

    Returns:
        rel_l2_arr:  (num_traj, num_steps)
        max_err_arr: (num_traj, num_steps)
    """
    validate_cc_pair(u_pred, v_pred, "prediction")
    validate_cc_pair(u_ref, v_ref, "reference")

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

    num_traj, num_steps = u_pred.shape[:2]

    rel_l2_arr = np.zeros((num_traj, num_steps), dtype=COMPUTE_DTYPE)
    max_err_arr = np.zeros((num_traj, num_steps), dtype=COMPUTE_DTYPE)

    for traj_idx in range(num_traj):
        for step in range(num_steps):
            rel_l2, max_err = compute_error_cell_center(
                u_pred[traj_idx, step],
                v_pred[traj_idx, step],
                u_ref[traj_idx, step],
                v_ref[traj_idx, step],
            )

            rel_l2_arr[traj_idx, step] = rel_l2
            max_err_arr[traj_idx, step] = max_err

    return rel_l2_arr, max_err_arr


def format_table(
    method_name: str,
    rel_l2_mean: np.ndarray,
    rel_l2_std: np.ndarray,
    max_err_mean: np.ndarray,
    max_err_std: np.ndarray,
) -> str:
    """
    格式化为与原 MAC 版本一致的表格字符串。
    """
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  Method: {method_name}")
    lines.append(f"{'=' * 60}")

    header = (
        f"{'Step':<8}"
        f"{'RelL2_mean':<14}"
        f"{'RelL2_std':<14}"
        f"{'MaxErr_mean':<14}"
        f"{'MaxErr_std':<14}"
    )

    lines.append(header)
    lines.append("-" * 60)

    for step in range(len(rel_l2_mean)):
        row = (
            f"{step + 1:<8}"
            f"{rel_l2_mean[step]:<14.4e}"
            f"{rel_l2_std[step]:<14.4e}"
            f"{max_err_mean[step]:<14.4e}"
            f"{max_err_std[step]:<14.4e}"
        )
        lines.append(row)

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    nc_path = Path(NC_FILE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_txt = output_dir / "cell_center_errors_float32.txt"

    if not nc_path.exists():
        raise FileNotFoundError(f"Input NetCDF file does not exist: {nc_path}")

    print(f"读取 CC 网格数据: {nc_path}")
    ds = xr.open_dataset(nc_path)

    try:
        ref_u_key, ref_v_key = pick_existing_pair(
            ds,
            REF_FIELD_CANDIDATES,
            name="reference",
        )

        method_map = build_method_map(ds)
        eval_methods = [m for m in METHOD_ORDER if m in method_map]

        if not eval_methods:
            raise RuntimeError("No methods from METHOD_ORDER were detected.")

        print(f"Reference variables: {ref_u_key}, {ref_v_key}")
        print("Detected methods:")
        for method in eval_methods:
            u_key, v_key = method_map[method]
            print(f"  {method}: {u_key}, {v_key}")

        u_ref = load_cc_array(ds, ref_u_key)
        v_ref = load_cc_array(ds, ref_v_key)
        validate_cc_pair(u_ref, v_ref, "reference")

        num_traj, num_steps, nx, ny = u_ref.shape

        print(f"轨迹数: {num_traj}, 步数: {num_steps}")
        print(f"CC grid: Nx={nx}, Ny={ny}")
        print(f"u_ref shape = {u_ref.shape}")
        print(f"v_ref shape = {v_ref.shape}")

        all_output_lines = [
            "Cell-Center Velocity Error Report",
            f"Input NetCDF: {nc_path}",
            f"Trajectories: {num_traj}   Steps: {num_steps}",
            f"Reference variables: {ref_u_key}, {ref_v_key}",
            f"CC grid: {nx} x {ny}",
            "",
            "Metrics:",
            "  RelL2 = sqrt(sum(du^2)+sum(dv^2)) / sqrt(sum(u_ref^2)+sum(v_ref^2))",
            "  MaxErr = max sqrt(du^2 + dv^2)",
            "",
        ]

        for method in eval_methods:
            u_key, v_key = method_map[method]

            print(f"  计算方法: {method} ...")

            u_pred = load_cc_array(ds, u_key)
            v_pred = load_cc_array(ds, v_key)
            validate_cc_pair(u_pred, v_pred, method)

            rel_l2_arr, max_err_arr = compute_errors_for_method(
                u_pred,
                v_pred,
                u_ref,
                v_ref,
            )

            # 跨 seed 求均值与标准差
            rel_l2_mean = rel_l2_arr.mean(axis=0, dtype=COMPUTE_DTYPE)
            rel_l2_std = rel_l2_arr.std(axis=0, dtype=COMPUTE_DTYPE)
            max_err_mean = max_err_arr.mean(axis=0, dtype=COMPUTE_DTYPE)
            max_err_std = max_err_arr.std(axis=0, dtype=COMPUTE_DTYPE)

            table = format_table(
                method,
                rel_l2_mean,
                rel_l2_std,
                max_err_mean,
                max_err_std,
            )

            print(table)
            all_output_lines.append(table)

    finally:
        ds.close()

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output_lines))

    print(f"\n结果已保存至: {output_txt}")


if __name__ == "__main__":
    main()