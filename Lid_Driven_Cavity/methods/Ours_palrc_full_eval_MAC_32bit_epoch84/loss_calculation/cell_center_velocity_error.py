"""
从 MAC 格式 NetCDF 文件中读取轨迹，插值到体心网格，
计算各方法相对 L2 误差与 L_infinity 误差，
按步骤对所有 seed 求均值与标准差，输出为 txt。
"""

import os
import numpy as np
import xarray as xr

COMPUTE_DTYPE = np.float32


# ============================================================
# 工具函数
# ============================================================

def mac_to_cell_center(u_mac, v_mac):
    """
    将MAC网格速度插值到体心网格
    u_mac: (Nx+1, Ny) → u_cc: (Nx, Ny)
    v_mac: (Nx, Ny+1) → v_cc: (Nx, Ny)
    """
    half = COMPUTE_DTYPE(0.5)
    u_cc = half * (u_mac[:-1, :] + u_mac[1:, :])
    v_cc = half * (v_mac[:, :-1] + v_mac[:, 1:])
    return u_cc, v_cc


def compute_error_cell_center(u_pred_cc, v_pred_cc, u_ref_cc, v_ref_cc):
    """
    在体心网格上计算误差
    返回: rel_l2, max_err
    """
    du = u_pred_cc - u_ref_cc
    dv = v_pred_cc - v_ref_cc

    # Relative L2 error
    sq_diff = np.sum(du**2, dtype=COMPUTE_DTYPE) + np.sum(dv**2, dtype=COMPUTE_DTYPE)
    sq_ref = (
        np.sum(u_ref_cc**2, dtype=COMPUTE_DTYPE)
        + np.sum(v_ref_cc**2, dtype=COMPUTE_DTYPE)
        + COMPUTE_DTYPE(1e-12)
    )
    rel_l2 = np.sqrt(sq_diff / sq_ref, dtype=COMPUTE_DTYPE)

    # Max error（向量模的最大值）
    error_magnitude = np.sqrt(du**2 + dv**2, dtype=COMPUTE_DTYPE)
    max_err = np.max(error_magnitude)

    return rel_l2, max_err


def compute_errors_for_method(u_pred, v_pred, u_ref, v_ref):
    """
    对形状 (num_traj, num_steps, ...) 的 MAC 场批量计算误差。
    返回 rel_l2_arr, max_err_arr，形状均为 (num_traj, num_steps)。
    """
    num_traj, num_steps = u_pred.shape[:2]
    rel_l2_arr = np.zeros((num_traj, num_steps), dtype=COMPUTE_DTYPE)
    max_err_arr = np.zeros((num_traj, num_steps), dtype=COMPUTE_DTYPE)

    for i in range(num_traj):
        for t in range(num_steps):
            u_p_cc, v_p_cc = mac_to_cell_center(u_pred[i, t], v_pred[i, t])
            u_r_cc, v_r_cc = mac_to_cell_center(u_ref[i, t],  v_ref[i, t])
            rl2, mx = compute_error_cell_center(u_p_cc, v_p_cc, u_r_cc, v_r_cc)
            rel_l2_arr[i, t] = rl2
            max_err_arr[i, t] = mx

    return rel_l2_arr, max_err_arr


def format_table(method_name, rel_l2_mean, rel_l2_std, max_err_mean, max_err_std):
    """格式化为与示例一致的表格字符串。"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Method: {method_name}")
    lines.append(f"{'='*60}")
    header = f"{'Step':<8}{'RelL2_mean':<14}{'RelL2_std':<14}{'MaxErr_mean':<14}{'MaxErr_std':<14}"
    lines.append(header)
    lines.append("-" * 60)
    for step in range(len(rel_l2_mean)):
        row = (
            f"{step+1:<8}"
            f"{rel_l2_mean[step]:<14.4e}"
            f"{rel_l2_std[step]:<14.4e}"
            f"{max_err_mean[step]:<14.4e}"
            f"{max_err_std[step]:<14.4e}"
        )
        lines.append(row)
    return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================

def main():
    NC_FILE   = "../data/full_trajectories_MAC_32bit.nc"
    OUTPUT_DIR = "../results/cell_center_velocity_error"
    OUTPUT_TXT = os.path.join(OUTPUT_DIR, "cell_center_errors_float32.txt")

    print(f"读取数据: {NC_FILE}")
    ds = xr.open_dataset(NC_FILE)

    # 读取各方法的 MAC 速度场，全部使用 float32 计算
    methods = {
        "AI  (FNO)": ("u_ai",     "v_ai"),
        "DNS-128":   ("u_dns128", "v_dns128"),
        "DNS-64":    ("u_dns64",  "v_dns64"),
    }
    u_ref = ds["u_ref"].values.astype(COMPUTE_DTYPE, copy=False)  # (traj, step, Nx+1, Ny)
    v_ref = ds["v_ref"].values.astype(COMPUTE_DTYPE, copy=False)  # (traj, step, Nx, Ny+1)

    num_traj, num_steps = u_ref.shape[:2]
    print(f"轨迹数: {num_traj}, 步数: {num_steps}")

    all_output_lines = [
        "Cell-Center Velocity Error Report",
        f"Trajectories: {num_traj}   Steps: {num_steps}",
        "Reference: DNS-256 downsampled to 64^2 (MAC)\n",
    ]

    for label, (u_key, v_key) in methods.items():
        print(f"  计算方法: {label} ...")
        u_pred = ds[u_key].values.astype(COMPUTE_DTYPE, copy=False)
        v_pred = ds[v_key].values.astype(COMPUTE_DTYPE, copy=False)

        rel_l2_arr, max_err_arr = compute_errors_for_method(u_pred, v_pred, u_ref, v_ref)

        # 跨 seed 求均值与标准差
        rel_l2_mean = rel_l2_arr.mean(axis=0, dtype=COMPUTE_DTYPE)
        rel_l2_std = rel_l2_arr.std(axis=0, dtype=COMPUTE_DTYPE)
        max_err_mean = max_err_arr.mean(axis=0, dtype=COMPUTE_DTYPE)
        max_err_std = max_err_arr.std(axis=0, dtype=COMPUTE_DTYPE)

        table = format_table(label, rel_l2_mean, rel_l2_std, max_err_mean, max_err_std)
        print(table)
        all_output_lines.append(table)

    ds.close()

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output_lines))
    print(f"\n结果已保存至: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
