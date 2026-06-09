import os
# Use GPU. Do NOT force CPU.
os.environ.pop("JAX_PLATFORMS", None)
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_deterministic_ops" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_deterministic_ops").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import sys
import time
import pickle
import types
import importlib
from functools import partial

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp


# =======================================================
# 0. Config
# =======================================================

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

try:
    GPU_DEVICES = jax.devices("gpu")
except RuntimeError:
    GPU_DEVICES = []

if len(GPU_DEVICES) == 0:
    print("❌ No JAX GPU backend detected.")
    print("Current JAX devices:", jax.devices())
    print("请确认安装 CUDA 版 jax/jaxlib，并且没有设置 JAX_PLATFORMS=cpu。")
    sys.exit(1)

print("✅ JAX GPU devices:", GPU_DEVICES)

L = 16.0
T = 4.0

N_HIGH = 4000
N_EVAL = 200

N_NATIVE_LIST = [200, 400, 800, 1000, 2000]

N_TIME_STEPS = 250
WENO_REF_SUBSTEPS = 40
NN_SUBSTEPS = 2

# Traditional solver substeps.
# Keep CFL approximately consistent as resolution increases.
SUBSTEPS_BY_NATIVE = {
    200: 2,
    400: 4,
    800: 8,
    1000: 10,
    2000: 20,
}

SEED_RANGE = range(1816, 1866)
SEEDS = list(SEED_RANGE)

NU_LIST = [0.0, 0.001, 0.005, 0.01]

SHARP_ALPHA = 0.5

# If GPU OOM, reduce to 5 or 1.
CHUNK_SIZE = 1

RUN_OURS = True

MODEL_MODULE_NAME = "model.cp_model"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH_CANDIDATES = [
    os.path.join(PACKAGE_DIR, "weights", "model_fixed.pkl"),
    os.path.join(SCRIPT_DIR, "model_fixed.pkl"),
]

OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs_rerun")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_CSV = os.path.join(
    OUTPUT_DIR,
    "full_raw_results_schemeC_n200_multinu_exact_N200_400_800_1000_2000_gpu.csv",
)
AGG_CSV = os.path.join(
    OUTPUT_DIR,
    "aggregated_results_schemeC_n200_multinu_exact_N200_400_800_1000_2000_gpu.csv",
)
AGG_FLAT_CSV = os.path.join(
    OUTPUT_DIR,
    "aggregated_results_schemeC_n200_multinu_exact_N200_400_800_1000_2000_gpu_flat.csv",
)
FIRST_SEED_NPZ = os.path.join(
    OUTPUT_DIR,
    "first_seed_bundle_schemeC_n200_multinu_exact_N200_400_800_1000_2000_gpu.npz",
)

SAVE_TRAJECTORIES = False
TRAJ_SAVE_DIR = os.path.join(
    OUTPUT_DIR,
    "trajectory_records_schemeC_n200_multinu_exact_N200_400_800_1000_2000_gpu",
)


# =======================================================
# 1. Pickle compatibility patch
# =======================================================

if "jax._src.device_array" not in sys.modules:
    mock_module = types.ModuleType("jax._src.device_array")
    mock_module.DeviceArray = jax.Array
    mock_module._DeviceArray = jax.Array
    sys.modules["jax._src.device_array"] = mock_module


# =======================================================
# 2. Import NN model
# =======================================================

try:
    import train

    model_module = importlib.import_module(MODEL_MODULE_NAME)
    my_cp_model = model_module.conv_fno_1d_forward

    train.model_forward = my_cp_model

    # IMPORTANT:
    # Do NOT patch train.down_func here.
    # Do NOT do:
    #     train.down_func = partial(train.downsample_cell_average, R=1)

    from train import multi_outer_steps_lc_forward

    jitted_nn_forward = jax.jit(
        multi_outer_steps_lc_forward,
        static_argnums=(3,),
        static_argnames=["substeps"],
    )

    print(f"✅ NN module imported: {MODEL_MODULE_NAME}")
    print("✅ train.down_func is NOT patched. Using train.py original logic.")

except Exception as e:
    RUN_OURS = False
    train = None
    jitted_nn_forward = None
    print(f"⚠️ NN module unavailable. Ours will be skipped. Error: {e}")


# =======================================================
# 3. Grid / forcing / restriction
# =======================================================

def resolve_model_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No valid checkpoint found. Tried:\n" + "\n".join(candidates))


def left_endpoint_grid(N, L):
    return jnp.linspace(0.0, L, N, endpoint=False)


def cell_average_source_grid(N_native, N_high, L):
    """
    Exact high-grid source coordinates grouped into native-resolution cells.

    This is the strong WENO-Z exact-restriction protocol:
        N_high -> N_native by reshape + mean.

    Requires:
        N_high % N_native == 0
    """
    assert N_high % N_native == 0, (
        f"N_HIGH={N_high} must be divisible by N_native={N_native}. "
        "Exact-restriction protocol cannot support this N_native."
    )

    R = N_high // N_native
    x_high = left_endpoint_grid(N_high, L)
    return x_high.reshape(N_native, R)


def forcing_pointwise(x, t, A, omega, phi, ell, L=16.0):
    """
    Pointwise forcing on a 1D grid.

    x: [N]
    A, omega, phi, ell: [J]
    return: [N]
    """
    phase = omega * t + (2.0 * jnp.pi * ell * x[:, None] / L) + phi
    return jnp.sum(A * jnp.sin(phase), axis=1)


def forcing_cell_average(x_blocks, t, A, omega, phi, ell, L=16.0):
    """
    Exact block-averaged forcing from high-grid blocks.

    x_blocks:
        [N_native, R]
    return:
        [N_native]
    """
    phase = (
        omega * t
        + (2.0 * jnp.pi * ell * x_blocks[:, :, None] / L)
        + phi
    )
    f_high_block = jnp.sum(A * jnp.sin(phase), axis=2)
    return jnp.mean(f_high_block, axis=1)


def forcing_source(x_source, t, A, omega, phi, ell, L=16.0):
    """
    Unified source wrapper.

    x_source.ndim == 1:
        pointwise forcing, used by high-resolution reference and Ours protocol.
    x_source.ndim == 2:
        exact block-averaged forcing, used by traditional methods.
    """
    if x_source.ndim == 1:
        return forcing_pointwise(x_source, t, A, omega, phi, ell, L=L)
    if x_source.ndim == 2:
        return forcing_cell_average(x_source, t, A, omega, phi, ell, L=L)
    raise ValueError(f"Unexpected x_source.ndim={x_source.ndim}")


def init_u0_batch_schemeC(A, omega, phi, ell, N, L):
    """
    High-grid pointwise initial condition.
    """
    x = left_endpoint_grid(N, L)

    def single(A_i, w_i, p_i, l_i):
        return forcing_pointwise(x, 0.0, A_i, w_i, p_i, l_i, L=L)

    return jax.vmap(single)(A, omega, phi, ell)


def flux(u):
    return 0.5 * u**2


def downsample_avg(u_fine, factor):
    """
    Forward block-average downsampling along the last axis.

    Supports:
        [N]
        [B, N]
        [T, N]
        [B, T, N]
    """
    if factor == 1:
        return u_fine

    shape = list(u_fine.shape)
    N_fine = shape[-1]

    assert N_fine % factor == 0, (
        f"N_fine={N_fine} is not divisible by factor={factor}"
    )

    N_coarse = N_fine // factor
    new_shape = shape[:-1] + [N_coarse, factor]
    return u_fine.reshape(new_shape).mean(axis=-1)


def to_numpy(a):
    return np.asarray(jax.device_get(a)) if not isinstance(a, np.ndarray) else a


# =======================================================
# 4. Numerical kernels
# =======================================================

def viscous_term(u, dx, nu):
    """
    JAX-safe viscous term.

    Do NOT use Python:
        if nu <= 0.0:
            ...
    because nu can be a tracer under jit/scan.
    """
    u_xx = (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dx**2
    return nu * u_xx


def weno_z_reconstruct(v):
    eps = 1e-8

    vm2 = jnp.roll(v, 2)
    vm1 = jnp.roll(v, 1)
    vp1 = jnp.roll(v, -1)
    vp2 = jnp.roll(v, -2)

    beta0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v) ** 2 + \
            (1.0 / 4.0) * (vm2 - 4.0 * vm1 + 3.0 * v) ** 2

    beta1 = (13.0 / 12.0) * (vm1 - 2.0 * v + vp1) ** 2 + \
            (1.0 / 4.0) * (vm1 - vp1) ** 2

    beta2 = (13.0 / 12.0) * (v - 2.0 * vp1 + vp2) ** 2 + \
            (1.0 / 4.0) * (3.0 * v - 4.0 * vp1 + vp2) ** 2

    tau5 = jnp.abs(beta0 - beta2)

    d0, d1, d2 = 0.1, 0.6, 0.3

    alpha0 = d0 * (1.0 + (tau5 / (beta0 + eps)) ** 2)
    alpha1 = d1 * (1.0 + (tau5 / (beta1 + eps)) ** 2)
    alpha2 = d2 * (1.0 + (tau5 / (beta2 + eps)) ** 2)

    wm_sum = alpha0 + alpha1 + alpha2

    p0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v) / 6.0
    p1 = (-vm1 + 5.0 * v + 2.0 * vp1) / 6.0
    p2 = (2.0 * v + 5.0 * vp1 - vp2) / 6.0

    return (alpha0 / wm_sum) * p0 + \
           (alpha1 / wm_sum) * p1 + \
           (alpha2 / wm_sum) * p2


def numerical_flux_wz(u):
    """
    Scheme-C WENO-Z global Lax-Friedrichs flux splitting.
    """
    a = jnp.max(jnp.abs(u))
    f = flux(u)

    fp = 0.5 * (f + a * u)
    fm = 0.5 * (f - a * u)

    Fp = weno_z_reconstruct(fp)

    # Keep original Scheme-C negative-flux reconstruction convention.
    Fm_flipped = weno_z_reconstruct(fm[::-1])[::-1]
    Fm = jnp.roll(Fm_flipped, -1)

    return Fp + Fm


def numerical_flux_fvm_rusanov(u):
    """
    First-order Rusanov flux.
    """
    f = flux(u)
    uR = jnp.roll(u, -1)
    fR = jnp.roll(f, -1)
    alpha = jnp.maximum(jnp.abs(u), jnp.abs(uR))

    return 0.5 * (f + fR) - 0.5 * alpha * (uR - u)


def rhs_wenoz(u, t, dx, x_source, A, omega, phi, ell, nu, L):
    F = numerical_flux_wz(u)
    dudt_flux = -(F - jnp.roll(F, 1)) / dx
    dudt_source = forcing_source(x_source, t, A, omega, phi, ell, L=L)
    return dudt_flux + viscous_term(u, dx, nu) + dudt_source


def rhs_fvm(u, t, dx, x_source, A, omega, phi, ell, nu, L):
    F = numerical_flux_fvm_rusanov(u)
    dudt_flux = -(F - jnp.roll(F, 1)) / dx
    dudt_source = forcing_source(x_source, t, A, omega, phi, ell, L=L)
    return dudt_flux + viscous_term(u, dx, nu) + dudt_source


# =======================================================
# 5. RK4 trajectory solvers
# =======================================================

def rk4_step(rhs_fn, u, t, dt, dx, x_source, A, omega, phi, ell, nu, L):
    k1 = rhs_fn(u, t, dx, x_source, A, omega, phi, ell, nu, L)
    k2 = rhs_fn(u + 0.5 * dt * k1, t + 0.5 * dt, dx, x_source, A, omega, phi, ell, nu, L)
    k3 = rhs_fn(u + 0.5 * dt * k2, t + 0.5 * dt, dx, x_source, A, omega, phi, ell, nu, L)
    k4 = rhs_fn(u + dt * k3, t + dt, dx, x_source, A, omega, phi, ell, nu, L)

    return u + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve_trajectory_generic(
    u0,
    A,
    omega,
    phi,
    ell,
    nu,
    L,
    T,
    N_time_steps,
    N_substeps,
    dx,
    x_source,
    downsample_factor,
    rhs_fn,
):
    dt_total = T / N_time_steps
    dt_sub = dt_total / N_substeps

    def scan_body(carry, _):
        u_curr, t_curr = carry

        def loop_body(i, val):
            u, t = val
            u_new = rk4_step(
                rhs_fn,
                u,
                t,
                dt_sub,
                dx,
                x_source,
                A,
                omega,
                phi,
                ell,
                nu,
                L,
            )
            return (u_new, t + dt_sub)

        u_next, t_next = jax.lax.fori_loop(
            0,
            N_substeps,
            loop_body,
            (u_curr, t_curr),
        )

        u_save = downsample_avg(u_next, downsample_factor)
        return (u_next, t_next), u_save

    _, trajectory = jax.lax.scan(
        scan_body,
        (u0, 0.0),
        None,
        length=N_time_steps,
    )

    return trajectory


@partial(jax.jit, static_argnames=["N_time_steps", "N_substeps", "downsample_factor"])
def solve_trajectory_wenoz(
    u0,
    A,
    omega,
    phi,
    ell,
    nu,
    L,
    T,
    N_time_steps,
    N_substeps,
    dx,
    x_source,
    downsample_factor=1,
):
    return solve_trajectory_generic(
        u0,
        A,
        omega,
        phi,
        ell,
        nu,
        L,
        T,
        N_time_steps,
        N_substeps,
        dx,
        x_source,
        downsample_factor,
        rhs_wenoz,
    )


@partial(jax.jit, static_argnames=["N_time_steps", "N_substeps", "downsample_factor"])
def solve_trajectory_fvm(
    u0,
    A,
    omega,
    phi,
    ell,
    nu,
    L,
    T,
    N_time_steps,
    N_substeps,
    dx,
    x_source,
    downsample_factor=1,
):
    return solve_trajectory_generic(
        u0,
        A,
        omega,
        phi,
        ell,
        nu,
        L,
        T,
        N_time_steps,
        N_substeps,
        dx,
        x_source,
        downsample_factor,
        rhs_fvm,
    )


batch_solve_wenoz = jax.vmap(
    solve_trajectory_wenoz,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
)

solve_batch_wenoz = jax.jit(
    batch_solve_wenoz,
    static_argnums=(8, 9, 12),
)


batch_solve_fvm = jax.vmap(
    solve_trajectory_fvm,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
)

solve_batch_fvm = jax.jit(
    batch_solve_fvm,
    static_argnums=(8, 9, 12),
)


# =======================================================
# 6. Random IC generation with Scheme-C key order
# =======================================================

def make_keys_from_seeds(seeds):
    keys = [jax.random.PRNGKey(int(s)) for s in seeds]
    return jnp.stack(keys, axis=0)


def sample_params_from_keys_schemeC(keys, J=5):
    """
    Match Scheme-C / training-data key order:
        k1,k2,k3,k4 = split(key, 4)
        A     <- k1
        omega <- k4
        phi   <- k3
        ell   <- k2
    """
    subkeys = jax.vmap(lambda k: jax.random.split(k, 4))(keys)

    k1 = subkeys[:, 0]
    k2 = subkeys[:, 1]
    k3 = subkeys[:, 2]
    k4 = subkeys[:, 3]

    A = jax.vmap(
        lambda k: jax.random.uniform(k, (J,), minval=-0.5, maxval=0.5)
    )(k1)

    omega = jax.vmap(
        lambda k: jax.random.uniform(k, (J,), minval=-0.4, maxval=0.4)
    )(k4)

    phi = jax.vmap(
        lambda k: jax.random.uniform(k, (J,), minval=0.0, maxval=2.0 * jnp.pi)
    )(k3)

    ell = jax.vmap(
        lambda k: jax.random.choice(
            k,
            jnp.array([1, 2, 3], dtype=jnp.float32),
            shape=(J,),
        )
    )(k2)

    return A, omega, phi, ell


# =======================================================
# 7. Metrics and saving
# =======================================================

def cell_average_to_target_grid_np(arr, N_target):
    arr = to_numpy(arr)
    N_source = arr.shape[-1]

    if N_source == N_target:
        return arr.astype(np.float32, copy=False)

    if N_source % N_target != 0:
        raise ValueError(
            f"N_source={N_source} is not divisible by N_target={N_target}."
        )

    factor = N_source // N_target
    new_shape = arr.shape[:-1] + (N_target, factor)
    return arr.reshape(new_shape).mean(axis=-1).astype(np.float32, copy=False)


def attach_initial_frame_np(u0, traj):
    u0_np = to_numpy(u0).reshape(1, -1).astype(np.float32, copy=False)
    traj_np = to_numpy(traj).astype(np.float32, copy=False)

    if traj_np.ndim != 2:
        raise ValueError(f"Expected traj shape [T,N], got {traj_np.shape}")

    return np.concatenate([u0_np, traj_np], axis=0).astype(np.float32, copy=False)


def centered_gradient_periodic_np(u, L):
    u = np.asarray(u)
    N = u.shape[-1]
    dx = L / N
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def compute_trajectory_metrics(
    pred_traj,
    ref_traj,
    L,
    T,
    N_eval=200,
    sharp_alpha=0.5,
    eps=1e-12,
):
    pred_eval = cell_average_to_target_grid_np(pred_traj, N_eval)
    ref_eval = cell_average_to_target_grid_np(ref_traj, N_eval)

    n_common = min(pred_eval.shape[0], ref_eval.shape[0])
    pred_eval = pred_eval[:n_common]
    ref_eval = ref_eval[:n_common]

    pred_final = pred_eval[-1]
    ref_final = ref_eval[-1]

    abs_diff_final = np.abs(pred_final - ref_final)

    l1 = float(np.mean(abs_diff_final))
    linf = float(np.max(abs_diff_final))

    grad_ref = centered_gradient_periodic_np(ref_final, L)
    G_ref = np.abs(grad_ref)

    G_max = float(np.max(G_ref))
    threshold = sharp_alpha * G_max

    if G_max <= eps:
        sharp_mask = np.ones_like(G_ref, dtype=bool)
    else:
        sharp_mask = G_ref > threshold
        if not np.any(sharp_mask):
            sharp_mask = np.zeros_like(G_ref, dtype=bool)
            sharp_mask[int(np.argmax(G_ref))] = True

    E_SG_T4 = float(np.mean(abs_diff_final[sharp_mask]))

    mean_pred = np.mean(pred_eval, axis=1)
    mean_ref = np.mean(ref_eval, axis=1)

    mean_drift = np.abs(mean_pred - mean_ref)
    E_mean_max_0_4 = float(np.max(mean_drift))

    argmax_idx = int(np.argmax(mean_drift))
    time_grid = np.linspace(0.0, T, n_common)

    return {
        "L1": l1,
        "Linf": linf,
        "E_SG_T4": E_SG_T4,
        "E_mean_max_0_4": E_mean_max_0_4,
        "E_mean_final_T4": float(mean_drift[-1]),
        "E_mean_argmax_time": float(time_grid[argmax_idx]),
        "SG_points": int(np.sum(sharp_mask)),
        "SG_fraction": float(np.mean(sharp_mask)),
        "SG_threshold": float(threshold),
        "final_abs_error_p95": float(np.percentile(abs_diff_final, 95)),
        "final_abs_error_p99": float(np.percentile(abs_diff_final, 99)),
        "pred_eval_traj": pred_eval.astype(np.float32, copy=False),
        "ref_eval_traj": ref_eval.astype(np.float32, copy=False),
        "mean_drift_traj": mean_drift.astype(np.float32, copy=False),
        "final_abs_error": abs_diff_final.astype(np.float32, copy=False),
        "sharp_mask": sharp_mask.astype(np.int8, copy=False),
    }


def safe_method_name(method_name):
    return (
        method_name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .replace("/", "_")
    )


def save_trajectory_bundle(traj_save_dir, nu, seed, traj_bundle):
    os.makedirs(traj_save_dir, exist_ok=True)
    nu_tag = str(nu).replace(".", "p")
    save_path = os.path.join(
        traj_save_dir,
        f"burgers_traj_nu{nu_tag}_seed{seed}.npz",
    )
    np.savez_compressed(save_path, **traj_bundle)
    return save_path


def flatten_agg_columns(df):
    out = df.copy()
    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in out.columns.to_flat_index()
    ]
    return out


def get_numeric_metric_columns(df, exclude_cols):
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def evaluate_and_record(
    all_results,
    traj_bundle,
    method_name,
    pred_traj_full,
    ref_traj_full,
    nu,
    seed,
    time_cost,
    L,
    T,
    N_eval,
    sharp_alpha,
    extra_info=None,
):
    metrics = compute_trajectory_metrics(
        pred_traj=pred_traj_full,
        ref_traj=ref_traj_full,
        L=L,
        T=T,
        N_eval=N_eval,
        sharp_alpha=sharp_alpha,
    )

    row = {
        "nu": nu,
        "seed": seed,
        "Method": method_name,
        "L1": metrics["L1"],
        "Linf": metrics["Linf"],
        "E_SG_T4": metrics["E_SG_T4"],
        "E_mean_max_0_4": metrics["E_mean_max_0_4"],
        "E_mean_final_T4": metrics["E_mean_final_T4"],
        "E_mean_argmax_time": metrics["E_mean_argmax_time"],
        "SG_points": metrics["SG_points"],
        "SG_fraction": metrics["SG_fraction"],
        "SG_threshold": metrics["SG_threshold"],
        "final_abs_error_p95": metrics["final_abs_error_p95"],
        "final_abs_error_p99": metrics["final_abs_error_p99"],
        "Time": time_cost,
    }

    if extra_info:
        row.update(extra_info)

    all_results.append(row)

    if SAVE_TRAJECTORIES:
        safe = safe_method_name(method_name)
        traj_bundle[f"{safe}_traj_eval{N_eval}"] = metrics["pred_eval_traj"]
        traj_bundle[f"{safe}_mean_drift"] = metrics["mean_drift_traj"]
        traj_bundle[f"{safe}_final_abs_error"] = metrics["final_abs_error"]
        traj_bundle[f"{safe}_sharp_mask"] = metrics["sharp_mask"]

        if f"Reference_traj_eval{N_eval}" not in traj_bundle:
            traj_bundle[f"Reference_traj_eval{N_eval}"] = metrics["ref_eval_traj"]


# =======================================================
# 8. Main evaluation
# =======================================================

def run_batch_experiments():
    print("\n=== Scheme-C N=200 multi-viscosity exact-restriction evaluation on GPU ===")
    print("Traditional methods: FVM/WENO-Z native N = 200, 400, 800, 1000, 2000")
    print("Evaluation target: all methods restricted to N_eval=200")
    print("Reference: WENO-Z(4000) restricted to N_eval=200")
    print("Ours: Ours(200), train.py original protocol")
    print(f"N_HIGH             = {N_HIGH}")
    print(f"N_EVAL             = {N_EVAL}")
    print(f"N_NATIVE_LIST      = {N_NATIVE_LIST}")
    print(f"N_TIME_STEPS       = {N_TIME_STEPS}")
    print(f"WENO_REF_SUBSTEPS  = {WENO_REF_SUBSTEPS}")
    print(f"SUBSTEPS_BY_NATIVE = {SUBSTEPS_BY_NATIVE}")
    print(f"NN_SUBSTEPS        = {NN_SUBSTEPS}")
    print(f"Seeds              = {SEED_RANGE.start} to {SEED_RANGE.stop - 1}")
    print(f"Nu list            = {NU_LIST}")
    print(f"CHUNK_SIZE         = {CHUNK_SIZE}")
    print(f"Save trajectories  = {SAVE_TRAJECTORIES}")

    model_path = resolve_model_path(MODEL_PATH_CANDIDATES)
    print(f"Model path         = {model_path}")

    if RUN_OURS and jitted_nn_forward is not None:
        with open(model_path, "rb") as f:
            nn_params = pickle.load(f)
        print("✅ N=200 NN checkpoint loaded.")
    else:
        nn_params = None
        print("⚠️ Ours skipped.")

    all_results = []
    first_seed_bundle = None
    first_seed_saved = False

    x_high = left_endpoint_grid(N_HIGH, L)
    dx_high = L / N_HIGH

    x_source_native = {
        N: cell_average_source_grid(N, N_HIGH, L)
        for N in N_NATIVE_LIST
    }

    print("\n>>> Native exact source grids:")
    for N in N_NATIVE_LIST:
        factor_native = N_HIGH // N
        print(
            f"  N={N}, x_source shape={tuple(x_source_native[N].shape)}, "
            f"restriction factor={factor_native}, "
            f"substeps={SUBSTEPS_BY_NATIVE[N]}"
        )

    # -----------------------------
    # JIT warm-up
    # -----------------------------
    print("\n>>> Warming up JIT on GPU...")

    warm_seeds = SEEDS[:CHUNK_SIZE]
    keys_w = make_keys_from_seeds(warm_seeds)
    A_w, omega_w, phi_w, ell_w = sample_params_from_keys_schemeC(keys_w)
    u0_high_w = init_u0_batch_schemeC(A_w, omega_w, phi_w, ell_w, N_HIGH, L)

    nu_w = NU_LIST[0]

    # Warm reference.
    _ = solve_batch_wenoz(
        u0_high_w,
        A_w,
        omega_w,
        phi_w,
        ell_w,
        nu_w,
        L,
        T,
        N_TIME_STEPS,
        WENO_REF_SUBSTEPS,
        dx_high,
        x_high,
        1,
    ).block_until_ready()

    # Warm traditional solvers at all native resolutions.
    for N_native in N_NATIVE_LIST:
        factor_native = N_HIGH // N_native
        u0_native_w = downsample_avg(u0_high_w, factor_native)

        dx_native = L / N_native
        xsrc_native = x_source_native[N_native]
        substeps_native = SUBSTEPS_BY_NATIVE[N_native]

        _ = solve_batch_fvm(
            u0_native_w,
            A_w,
            omega_w,
            phi_w,
            ell_w,
            nu_w,
            L,
            T,
            N_TIME_STEPS,
            substeps_native,
            dx_native,
            xsrc_native,
            1,
        ).block_until_ready()

        _ = solve_batch_wenoz(
            u0_native_w,
            A_w,
            omega_w,
            phi_w,
            ell_w,
            nu_w,
            L,
            T,
            N_TIME_STEPS,
            substeps_native,
            dx_native,
            xsrc_native,
            1,
        ).block_until_ready()

    # Warm Ours.
    if nn_params is not None and jitted_nn_forward is not None:
        R_200 = N_HIGH // N_EVAL
        u0_ours_w = downsample_avg(u0_high_w, R_200)

        nu_batched_w = jnp.full((len(warm_seeds),), nu_w)
        dx_low_w = L / N_EVAL
        x_low_w = left_endpoint_grid(N_EVAL, L)
        f_params_w = (A_w, omega_w, ell_w, phi_w)
        phys_params_w = (nu_batched_w, dx_low_w, x_low_w, f_params_w, L)

        init_u_nn_w = u0_ours_w[..., None]
        dt_frame_w = T / N_TIME_STEPS

        pred_w = jitted_nn_forward(
            nn_params,
            init_u_nn_w,
            0.0,
            N_TIME_STEPS,
            dt_frame_w,
            phys_params_w,
            jax.random.PRNGKey(32),
            substeps=NN_SUBSTEPS,
        )
        pred_w.block_until_ready()

    print(">>> Warm-up finished.\n")

    total_chunks = len(NU_LIST) * int(np.ceil(len(SEEDS) / CHUNK_SIZE))
    chunk_counter = 0

    # -----------------------------
    # Main evaluation
    # -----------------------------
    for nu in NU_LIST:
        print(f"\n--- Processing nu = {nu} ---")

        for chunk_start in range(0, len(SEEDS), CHUNK_SIZE):
            chunk_counter += 1
            chunk_seeds = SEEDS[chunk_start:chunk_start + CHUNK_SIZE]
            B = len(chunk_seeds)

            print(
                f"\n>>> Chunk {chunk_counter}/{total_chunks}: "
                f"nu={nu}, seeds {chunk_seeds[0]}-{chunk_seeds[-1]}, B={B}"
            )

            keys = make_keys_from_seeds(chunk_seeds)
            A, omega, phi, ell = sample_params_from_keys_schemeC(keys)
            u0_high = init_u0_batch_schemeC(A, omega, phi, ell, N_HIGH, L)

            # -------------------------------------------
            # Reference WENO-Z N=4000
            # -------------------------------------------
            t0 = time.time()

            traj_ref_high = solve_batch_wenoz(
                u0_high,
                A,
                omega,
                phi,
                ell,
                nu,
                L,
                T,
                N_TIME_STEPS,
                WENO_REF_SUBSTEPS,
                dx_high,
                x_high,
                1,
            )
            traj_ref_high.block_until_ready()

            ref_time_total = time.time() - t0

            assert traj_ref_high.shape == (B, N_TIME_STEPS, N_HIGH), traj_ref_high.shape

            # Native initial states by exact restriction 4000 -> N_native.
            u0_native = {
                N: downsample_avg(u0_high, N_HIGH // N)
                for N in N_NATIVE_LIST
            }

            # -------------------------------------------
            # Ours prediction, Ours(200)
            # -------------------------------------------
            if nn_params is not None and jitted_nn_forward is not None:
                u0_ours = u0_native[N_EVAL]

                nu_batched = jnp.full((B,), nu)
                dx_low = L / N_EVAL
                x_low = left_endpoint_grid(N_EVAL, L)
                f_params = (A, omega, ell, phi)  # train.py convention
                phys_params = (nu_batched, dx_low, x_low, f_params, L)

                init_u_nn = u0_ours[..., None]
                dt_frame = T / N_TIME_STEPS

                t0 = time.time()

                pred_nn = jitted_nn_forward(
                    nn_params,
                    init_u_nn,
                    0.0,
                    N_TIME_STEPS,
                    dt_frame,
                    phys_params,
                    jax.random.PRNGKey(32),
                    substeps=NN_SUBSTEPS,
                )

                if pred_nn.ndim == 4 and pred_nn.shape[-1] == 1:
                    pred_nn = pred_nn[..., 0]

                pred_nn.block_until_ready()
                ours_time_total = time.time() - t0
            else:
                u0_ours = None
                pred_nn = None
                ours_time_total = np.nan

            # -------------------------------------------
            # Traditional predictions
            # -------------------------------------------
            trad_results = {}

            for N_native in N_NATIVE_LIST:
                u0_n = u0_native[N_native]
                dx_n = L / N_native
                xsrc_n = x_source_native[N_native]
                substeps_n = SUBSTEPS_BY_NATIVE[N_native]

                # FVM at native N.
                t0 = time.time()

                traj_fvm_n = solve_batch_fvm(
                    u0_n,
                    A,
                    omega,
                    phi,
                    ell,
                    nu,
                    L,
                    T,
                    N_TIME_STEPS,
                    substeps_n,
                    dx_n,
                    xsrc_n,
                    1,
                )
                traj_fvm_n.block_until_ready()

                trad_results[f"FVM ({N_native})"] = (
                    traj_fvm_n,
                    time.time() - t0,
                    N_native,
                    substeps_n,
                )

                # WENO-Z at native N.
                t0 = time.time()

                traj_wz_n = solve_batch_wenoz(
                    u0_n,
                    A,
                    omega,
                    phi,
                    ell,
                    nu,
                    L,
                    T,
                    N_TIME_STEPS,
                    substeps_n,
                    dx_n,
                    xsrc_n,
                    1,
                )
                traj_wz_n.block_until_ready()

                trad_results[f"WENO-Z ({N_native})"] = (
                    traj_wz_n,
                    time.time() - t0,
                    N_native,
                    substeps_n,
                )

            # Move arrays to CPU for metrics.
            u0_high_np = to_numpy(u0_high)
            traj_ref_high_np = to_numpy(traj_ref_high)

            u0_native_np = {
                N: to_numpy(arr)
                for N, arr in u0_native.items()
            }

            u0_ours_np = to_numpy(u0_ours) if u0_ours is not None else None
            pred_nn_np = to_numpy(pred_nn) if pred_nn is not None else None

            trad_results_np = {
                name: (to_numpy(traj), elapsed, N_native, substeps_n)
                for name, (traj, elapsed, N_native, substeps_n) in trad_results.items()
            }

            # -------------------------------------------
            # Per-seed metrics
            # -------------------------------------------
            for b, seed in enumerate(chunk_seeds):
                ref_traj_full = attach_initial_frame_np(
                    u0_high_np[b],
                    traj_ref_high_np[b],
                )

                traj_bundle = {
                    "nu": np.asarray(nu),
                    "seed": np.asarray(seed),
                    "N_high": np.asarray(N_HIGH),
                    "N_eval": np.asarray(N_EVAL),
                    "L": np.asarray(L),
                    "T": np.asarray(T),
                    "A": to_numpy(A[b]),
                    "omega": to_numpy(omega[b]),
                    "phi": to_numpy(phi[b]),
                    "ell": to_numpy(ell[b]),
                    "u0_high": u0_high_np[b],
                    "ref_time_per_seed": np.asarray(ref_time_total / B),
                }

                # Traditional methods.
                for method_name, (traj_np, elapsed_total, N_native, substeps_n) in trad_results_np.items():
                    pred_full = attach_initial_frame_np(
                        u0_native_np[N_native][b],
                        traj_np[b],
                    )

                    evaluate_and_record(
                        all_results=all_results,
                        traj_bundle=traj_bundle,
                        method_name=method_name,
                        pred_traj_full=pred_full,
                        ref_traj_full=ref_traj_full,
                        nu=nu,
                        seed=seed,
                        time_cost=elapsed_total / B,
                        L=L,
                        T=T,
                        N_eval=N_EVAL,
                        sharp_alpha=SHARP_ALPHA,
                        extra_info={
                            "N_native": N_native,
                            "N_eval": N_EVAL,
                            "native_substeps": substeps_n,
                            "traditional_substeps": substeps_n,
                            "nn_substeps": 0,
                            "source_treatment": f"exact_restriction_from_Nhigh{N_HIGH}",
                            "RefTime": ref_time_total / B,
                        },
                    )

                # Ours.
                if pred_nn_np is not None:
                    ours_full = attach_initial_frame_np(
                        u0_ours_np[b],
                        pred_nn_np[b],
                    )

                    evaluate_and_record(
                        all_results=all_results,
                        traj_bundle=traj_bundle,
                        method_name="Ours (200)",
                        pred_traj_full=ours_full,
                        ref_traj_full=ref_traj_full,
                        nu=nu,
                        seed=seed,
                        time_cost=ours_time_total / B,
                        L=L,
                        T=T,
                        N_eval=N_EVAL,
                        sharp_alpha=SHARP_ALPHA,
                        extra_info={
                            "N_native": N_EVAL,
                            "N_eval": N_EVAL,
                            "native_substeps": NN_SUBSTEPS,
                            "traditional_substeps": 0,
                            "nn_substeps": NN_SUBSTEPS,
                            "source_treatment": "train_py_original_ours_protocol",
                            "RefTime": ref_time_total / B,
                            "model_path": model_path,
                        },
                    )

                if SAVE_TRAJECTORIES:
                    save_path = save_trajectory_bundle(
                        TRAJ_SAVE_DIR,
                        nu,
                        seed,
                        traj_bundle,
                    )
                    if (seed - SEED_RANGE.start + 1) % 10 == 0:
                        print(f"Saved trajectory bundle: {save_path}")

                if not first_seed_saved:
                    first_seed_bundle = traj_bundle
                    first_seed_saved = True

            # Free big arrays explicitly.
            del traj_ref_high
            for name, (traj, _, _, _) in trad_results.items():
                del traj
            if pred_nn is not None:
                del pred_nn

    # -----------------------------
    # Statistics
    # -----------------------------
    df = pd.DataFrame(all_results)

    metric_cols = get_numeric_metric_columns(
        df,
        exclude_cols={"nu", "seed", "Method", "source_treatment", "model_path"},
    )

    agg_df = (
        df.groupby(["nu", "Method"])[metric_cols]
        .agg(["mean", "std", "min", "max", "median"])
    )

    agg_flat_df = flatten_agg_columns(agg_df).reset_index()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2400)
    pd.set_option("display.float_format", lambda x: "%.6e" % x)

    print("\n=== Data Collection Complete. Calculating Statistics... ===")
    print("\n>>> Aggregated Results:")

    show_cols = [
        "nu",
        "Method",
        "N_native_mean",
        "native_substeps_mean",
        "L1_mean",
        "Linf_mean",
        "E_SG_T4_mean",
        "E_mean_max_0_4_mean",
        "Time_mean",
        "RefTime_mean",
    ]
    show_cols = [c for c in show_cols if c in agg_flat_df.columns]
    print(agg_flat_df[show_cols].to_string(index=False))

    print("\n>>> Aggregated Results, full flattened:")
    print(agg_flat_df.to_string(index=False))

    df.to_csv(RAW_CSV, index=False)
    agg_df.to_csv(AGG_CSV)
    agg_flat_df.to_csv(AGG_FLAT_CSV, index=False)

    if first_seed_bundle is not None:
        np.savez_compressed(FIRST_SEED_NPZ, **first_seed_bundle)

    print(f"\nRaw results saved to: {RAW_CSV}")
    print(f"Aggregated results saved to: {AGG_CSV}")
    print(f"Aggregated flat results saved to: {AGG_FLAT_CSV}")
    print(f"First seed bundle saved to: {FIRST_SEED_NPZ}")

    if SAVE_TRAJECTORIES:
        print(f"Trajectory bundles saved to: {TRAJ_SAVE_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    run_batch_experiments()
