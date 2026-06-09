import os
import glob
import time
import random
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import xarray as xr
import jax.image as image
from loguru import logger

# ============================================================
# User Imports
# ============================================================
from model.conv_fno_model import model_forward 

# ============================================================
# 0. 配置与超参数
# ============================================================

# --- 物理步长设置 ---
# 如果数据 dt=0.01, PHYSICS_SUBSTEPS=10, 则物理求解器 dt=0.001
PHYSICS_SUBSTEPS = 2

TRAIN_DIR = "./train_burgers_data" 
# 注意：验证集将手动指定为 traj_0199.nc
VALID_FILE_PATTERN = "*0199.nc" 

SAVE_DIR = "./saved_models_forward_5_3e-5_new_loss"

PRED_STEPS = 5       # 预测未来帧数
TIME_STRIDE = 1      # 训练采样步长
FILE_BATCH_SIZE = 10 # 每个 NC 文件读取的 Batch 大小
TOTAL_TIME = 250     

EPOCHS = 75          
LEARNING_RATE =3*(1e-5)

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# 日志配置
LOG_FILE = "train_log_forward_5.log"
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(LOG_FILE, rotation="10 MB", level="DEBUG")

# ============================================================
# 1. 物理内核 (Physics Kernels)
# ============================================================

def forcing(x, t, A, omega, phi, ell, L):
    # 注意：此函数在 batch_advance_fn 中被 vmap 调用，因此处理的是单个样本
    # x: (N,) (shared grid, in_axes=None)
    # A, omega, phi, ell: (J,) (single sample params, in_axes=0)
    # t: scalar (single sample time, in_axes=0)
    
    # 广播逻辑：
    # x[:, None] -> shape (N, 1)
    # ell        -> shape (J,) 自动广播为 (1, J)
    # 运算结果 phase -> shape (N, J)
    phase = omega * t + (2 * jnp.pi * ell * x[:, None] / L) + phi
    
    # 在 mode 维度 J (axis=-1) 上求和，得到 (N,)
    return jnp.sum(A * jnp.sin(phase), axis=-1)

def flux(u):
    return 0.5 * u**2

def rusanov_flux(u):
    f = flux(u)
    uR = jnp.roll(u, -1)
    fR = jnp.roll(f, -1)
    a = jnp.maximum(jnp.abs(u), jnp.abs(uR))
    F = 0.5 * (f + fR) - 0.5 * a * (uR - u)
    return F

def viscous_term(u, dx, nu):
    u_xx = (jnp.roll(u, -1) - 2*u + jnp.roll(u, 1)) / dx**2
    return nu * u_xx

def rhs(u, t, dx, x, A, omega, phi, ell, nu, L):
    F = rusanov_flux(u)
    dudt_flux = -(F - jnp.roll(F, 1)) / dx
    dudt_visc = viscous_term(u, dx, nu)
    dudt_forcing = forcing(x, t, A, omega, phi, ell, L)
    return dudt_flux + dudt_visc + dudt_forcing

def rk4_step(u, t, dt, dx, x, A, omega, phi, ell, nu, L):
    k1 = rhs(u,              t,          dx, x, A, omega, phi, ell, nu, L)
    k2 = rhs(u + 0.5*dt*k1,  t+0.5*dt,   dx, x, A, omega, phi, ell, nu, L)
    k3 = rhs(u + 0.5*dt*k2,  t+0.5*dt,   dx, x, A, omega, phi, ell, nu, L)
    k4 = rhs(u + dt*k3,      t+dt,       dx, x, A, omega, phi, ell, nu, L)
    return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def advance_big_step(u, t0, dt_small, dx, x, A, omega, phi, ell, nu, L, small_steps):
    """
    单条轨迹向前推进一步（包含 small_steps 个小步）
    """
    def body(carry, _):
        u_curr, t_curr = carry
        u_next = rk4_step(u_curr, t_curr, dt_small, dx, x, A, omega, phi, ell, nu, L)
        return (u_next, t_curr + dt_small), None

    (u_final, t_final), _ = jax.lax.scan(
        body, 
        (u, t0), 
        None, 
        length=small_steps
    )
    return u_final, t_final

# 辅助函数：降采样与上采样
def downsample_cell_average(u_high, R):
    """(B, N_high, C) -> (B, N_low, C)"""
    B, N, C = u_high.shape
    
    # 增加维度检查，防止 reshape 错误
    if N % R != 0:
        raise ValueError(f"High res grid size {N} is not divisible by downsample factor {R}")
        
    return jnp.mean(u_high.reshape(B, -1, R, C), axis=2, keepdims=False)

def get_upsample_fn_1d(size_finer: int, method: str = 'linear'):
    if method == 'linear':
        def upsample_func(x):
            B, L, C = x.shape
            return image.resize(x, shape=(B, size_finer, C), method='linear')
        return upsample_func
    else:
        raise ValueError(f'Method {method} is not implemented for 1D!')

# 全局配置
GLOBAL_R = 2
#TODO 这里进行了修改 后面如果有bug 直接设置为200
GLOBAL_FINE_RES = 200
down_func = partial(downsample_cell_average, R=GLOBAL_R)
upsample_func = get_upsample_fn_1d(size_finer=GLOBAL_FINE_RES)

# ============================================================
# 2. 核心逻辑：Hybrid Solver Forward
# ============================================================

# 向量化 advance_big_step 以支持 Batch
batch_advance_fn = jax.vmap(
    advance_big_step,
    in_axes=(0, 0, None, None, None, 0, 0, 0, 0, 0, None, None)
)

def multi_outer_steps_lc_forward(nn_params, u_init, t_init, multi_steps, dt, phys_params, key, substeps):
    """
    混合求解器前向传播 (Hybrid Solver Forward Pass)
    """
    
    # 1. 准备物理微步长
    dt_small = dt / substeps
    
    # 2. 准备 Scan 参数
    scan_keys = jax.random.split(key, multi_steps)

    # 扩展 t_init 到 (Batch, 1)
    B = u_init.shape[0]
    if jnp.ndim(t_init) == 0:
        t_init_b = jnp.full((B, 1), t_init)
    else:
        t_init_b = jnp.reshape(t_init, (B, 1))

    # -----------------------------------------------------------
    # 优化：在循环外预计算所有不变参数
    # -----------------------------------------------------------
    # A. 解包物理参数 (Batch, J) or Scalar
    nu, dx_fine, x_grid_fine, f_params, L = phys_params
    
    # 解包 Forcing 参数: 确保顺序与 prepare_training_data 打包顺序一致 (A, omega, ell, phi)
    A, omega, ell_param, phi = f_params 
    
    # B. 预计算粗网格信息
    # 我们使用 u_init 来推断网格大小，因为网格在时间演化中是不变的
    u_init_down = down_func(u_init)
    
    high_res_L = u_init.shape[1]
    low_res_L = u_init_down.shape[1]
    scale_factor = high_res_L // low_res_L
    
    # 计算粗网格步长和坐标
    dx_coarse = dx_fine * scale_factor
    x_grid_coarse = x_grid_fine[::scale_factor]

    @jax.checkpoint 
    def step_fn(carry, curr_key):
        # carry 包含 (当前状态, 当前绝对时间)
        u_prev, t_prev = carry
        
        # -----------------------------------------------------------
        # A. 降采样 (Downsample): Fine -> Coarse
        # -----------------------------------------------------------
        u_prev_down = down_func(u_prev) # (B, N_coarse, C)
        
        # -----------------------------------------------------------
        # B. 粗网格物理求解 (Coarse Physics Solver)
        # -----------------------------------------------------------
        u_coarse_in = u_prev_down[..., 0] 
        t_prev_flat = jnp.reshape(t_prev, (-1,))

        # 调用 FVM + Rusanov 求解器 (vmap version)
        # 注意参数顺序需与 advance_big_step 定义一致: A, omega, phi, ell, nu, L
        u_next_weno, t_next_flat = batch_advance_fn(
            u_coarse_in,    # u
            t_prev_flat,    # t0
            dt_small,       # dt_small (scalar)
            dx_coarse,      # dx (scalar)
            x_grid_coarse,  # x (N_coarse,)
            A, omega, phi, ell_param, # Forcing params (Batch, J)
            nu,             # nu (Batch,)
            L,              # L (scalar)
            substeps        # loops (static int)
        )
        
        # 恢复维度
        u_next_weno = u_next_weno[..., None]
        t_next = jnp.reshape(t_next_flat, (-1, 1))

        # -----------------------------------------------------------
        # C. 上采样 (Upsample): Coarse -> Fine
        # -----------------------------------------------------------
        u_next_upsample = upsample_func(u_next_weno) 
        
        # -----------------------------------------------------------
        # D. AI 修正 (AI Correction)
        # -----------------------------------------------------------
        u_input = jnp.concatenate([u_prev, u_next_upsample], axis=-1)
        u_correction = model_forward.apply(nn_params, curr_key, u_input)

        # =======================================================
        # Mean-preserving constraint
        # =======================================================
        # Enforce zero spatial mean for the learned correction at every rollout step:
        #     mean_x(u_correction) = 0
        # Hence:
        #     mean_x(u_next_final) = mean_x(u_next_upsample)
        #
        # This is a step-wise constraint, not a final-time post-processing.
        u_correction = u_correction - jnp.mean(u_correction, axis=1, keepdims=True)
        u_next_final = u_next_upsample + u_correction


        # u_next_final = u_next_upsample + u_correction
        
        return (u_next_final, t_next), u_next_final

    # 3. Run Scan
    (final_u, final_t), trajectory = jax.lax.scan(step_fn, (u_init, t_init_b), scan_keys)
    
    # 调整输出维度: (Steps, Batch, L, C) -> (Batch, Steps, L, C)
    trajectory = jnp.transpose(trajectory, (1, 0, 2, 3))
    
    return trajectory

def compute_loss(nn_params, u_init_raw, target_traj, dt, phys_params, t_init, key, substeps):
    if u_init_raw.ndim == 2:
        u_init = u_init_raw[..., None]
    else:
        u_init = u_init_raw

    multi_steps = target_traj.shape[1]

    # 调用 Forward 时不再需要 ts，物理时间由 t_init + dt 自动推演
    pred_traj = multi_outer_steps_lc_forward(
        nn_params, 
        u_init, 
        t_init,
        multi_steps, 
        dt, 
        phys_params, 
        key,
        substeps=substeps
    )

    pred_traj_squeezed = jnp.squeeze(pred_traj, axis=-1)
    if target_traj.ndim == 4:
         target_traj = jnp.squeeze(target_traj, axis=-1)

    diff = pred_traj_squeezed - target_traj
    loss_1 = jnp.mean(jnp.abs(diff))
    loss_2 = jnp.sqrt(jnp.mean(diff**2))
    
    return 0.4 * loss_1 + 0.6 * loss_2
