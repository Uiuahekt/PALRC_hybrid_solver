import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional
import time

# ==============================================================================
# 1. 物理特征提取器 (Updated with your custom features)
# ==============================================================================

def gen_features_1d(x: jnp.ndarray, length: float = 16.0) -> jnp.ndarray:
    """
    计算增强版 1D 物理特征 (包含激波捕捉算子、积分量等)。
    
    Args:
        x: Input array [Batch, Grid, Channels]. Assumes channel 0 is u.
        length: Physical domain length.
        
    Returns:
        features: [Batch, Grid, 10]
    """
    # 自动推断 grid_size
    B, grid_size, _ = x.shape
    
    u = x[:, :, 0]
    dx = length / grid_size
    
    # 1. 一阶导数 (Central Difference)
    du = (jnp.roll(u, -1, 1) - jnp.roll(u, 1, 1)) / (2 * dx)
    
    # 2. 二阶导数 (Laplacian)
    d2u = (jnp.roll(u, -1, 1) - 2 * u + jnp.roll(u, 1, 1)) / (dx**2)
    
    # 3. 局部高频残差 (Local Fluctuation / High-pass filter)
    # (Left + Right)/2 - Center
    udiff = (jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1))/2 - u
    
    # 4. 归一化累积积分 (Normalized Cumulative Integral)
    # 帮助模型感知全局质量分布
    integ = jnp.cumsum(u, 1) * dx
    integ = (integ - jnp.mean(integ, 1, keepdims=True)) / (jnp.std(integ, 1, keepdims=True) + 1e-6)
    
    # 5. 归一化坐标 (0 to 1)
    coord = jnp.tile(jnp.linspace(0, 1, grid_size)[None, :], (B, 1))
    
    # --- 组合所有特征 (共 10 个通道) ---
    features = jnp.stack([
        u,              # 0. 原始速度
        du,             # 1. 梯度
        jnp.tanh(du),   # 2. 软激波检测器 (Gradient clipper/Shock sensor)
        d2u,            # 3. 粘性项/曲率
        udiff,          # 4. 高频细节
        u * du,         # 5. 对流非线性项 (Advection)
        integ,          # 6. 全局积分特征
        0.5 * u**2,     # 7. 动能 (Energy)
        jnp.abs(du),    # 8. 梯度幅值 (Total Variation proxy)
        coord           # 9. 空间坐标
    ], axis=-1)
    
    return features

# ==============================================================================
# 2. 极速 UNet (FastUNet 1D) - 保持不变
# ==============================================================================

class FastUNet1D(hk.Module):
    def __init__(self, base_channels=32, output_channels=64, name=None):
        super().__init__(name=name)
        self.base = base_channels
        self.output_channels = output_channels

    def __call__(self, x):
        # Encoder
        e1 = hk.Conv1D(self.base, 3, stride=1, padding='SAME', name='enc_1')(x)
        e1 = jax.nn.gelu(e1)

        e2 = hk.Conv1D(self.base*2, 3, stride=2, padding='SAME', name='enc_2')(e1)
        e2 = jax.nn.gelu(e2)

        e3 = hk.Conv1D(self.base*4, 3, stride=2, padding='SAME', name='enc_3')(e2)
        e3 = jax.nn.gelu(e3)

        # Bottleneck
        b = hk.Conv1D(self.base*4, 3, stride=1, padding='SAME', name='bottleneck')(e3)
        b = jax.nn.gelu(b)

        # Decoder
        d2 = hk.Conv1DTranspose(self.base*2, 2, stride=2, padding='SAME', name='up_1')(b)
        d2 = jax.nn.gelu(d2)
        if d2.shape[1] != e2.shape[1]: d2 = d2[:, :e2.shape[1], :]
        d2 = jnp.concatenate([d2, e2], axis=-1)
        d2 = hk.Conv1D(self.base*2, 3, stride=1, padding='SAME', name='dec_1')(d2)
        d2 = jax.nn.gelu(d2)

        d1 = hk.Conv1DTranspose(self.base, 2, stride=2, padding='SAME', name='up_2')(d2)
        d1 = jax.nn.gelu(d1)
        if d1.shape[1] != e1.shape[1]: d1 = d1[:, :e1.shape[1], :]
        d1 = jnp.concatenate([d1, e1], axis=-1)
        d1 = hk.Conv1D(self.base, 3, stride=1, padding='SAME', name='dec_2')(d1)
        d1 = jax.nn.gelu(d1)

        out = hk.Conv1D(self.output_channels, 1, stride=1, padding='SAME', name='head')(d1)
        return out

# ==============================================================================
# 3. 多层 CP 分解谱卷积 (1D) - 保持不变
# ==============================================================================

class MultiLayerCPSpectralConv1d(hk.Module):
    def __init__(self, n_layers, in_channels, out_channels, modes_x, rank=64, name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.rank = rank

    def get_complex_factor(self, shape, name):
        scale = 1.0 / (self.rank ** 0.5)
        real = hk.get_parameter(f"{name}_real", shape, init=hk.initializers.RandomNormal(stddev=scale))
        imag = hk.get_parameter(f"{name}_imag", shape, init=hk.initializers.RandomNormal(stddev=scale))
        return real + 1j * imag

    def get_complex_weight(self, shape, name):
        real = hk.get_parameter(f"{name}_real", shape, init=hk.initializers.Constant(1.0))
        imag = hk.get_parameter(f"{name}_imag", shape, init=hk.initializers.Constant(0.0))
        return real + 1j * imag

    def __call__(self, x, layer_index):
        x = jnp.transpose(x, (0, 2, 1))
        B, C, W = x.shape
        x_ft = jnp.fft.rfft(x, norm='ortho')
        ft_kept = x_ft[:, :, :self.modes_x]
        
        weights = self.get_complex_weight((self.rank,), "cp_weights")
        u_layer = self.get_complex_factor((self.n_layers, self.rank), "u_layer")[layer_index]
        u_in = self.get_complex_factor((self.in_channels, self.rank), "u_in")
        u_out = self.get_complex_factor((self.out_channels, self.rank), "u_out")
        u_x = self.get_complex_factor((self.modes_x, self.rank), "u_x")

        eff_weights = weights * u_layer
        term = jnp.einsum('bix,ir->brx', ft_kept, u_in)
        term = term * eff_weights[None, :, None]
        term = jnp.einsum('brx,xr->brx', term, u_x)
        out_kept = jnp.einsum('brx,or->box', term, u_out)

        out_ft = jnp.zeros((B, self.out_channels, W//2+1), dtype=x_ft.dtype)
        out_ft = out_ft.at[:, :, :self.modes_x].set(out_kept)
        x_out = jnp.fft.irfft(out_ft, n=W, norm='ortho')
        return jnp.transpose(x_out, (0, 2, 1))

# ==============================================================================
# 4. 顶层封装 (Top-Level Model 1D)
# ==============================================================================

class FastUNetFNO1D(hk.Module):
    def __init__(self, 
                 in_channels=2,        # 原始输入通道 (u, x)
                 unet_base=32,         
                 unet_out=64,          
                 fno_width=64,         
                 fno_modes=24,         
                 fno_layers=4,         
                 fno_rank=64,          
                 domain_length=16.0,
                 name=None):
        super().__init__(name=name)
        self.domain_length = domain_length
        self.fno_layers = fno_layers
        
        # 1. UNet 1D
        self.unet = FastUNet1D(base_channels=unet_base, output_channels=unet_out)
        
        # 2. FNO Projection
        # 输入维度 = UNet输出(64) + 手工特征(10) = 74
        self.fno_in = hk.Linear(fno_width)
        
        # 3. CP Spectral Block
        self.cp_conv = MultiLayerCPSpectralConv1d(
            n_layers=fno_layers,
            in_channels=fno_width,
            out_channels=fno_width,
            modes_x=fno_modes,
            rank=fno_rank
        )
        
        # 4. Output Heads
        self.fno_out_hidden = hk.Linear(128)
        self.fno_out = hk.Linear(1) # 只输出 u

    def __call__(self, x: jnp.ndarray):
        """
        x: [B, Grid, C] (通常 C=2: u, grid_x)
        """
        # --- A. UNet 提取多尺度特征 ---
        unet_feat = self.unet(x) # [B, Grid, 64]
        
        # --- B. 手工物理特征 ---
        # 你的新特征提取器已经包含了 u, coord, du, d2u 等 10 个特征
        # 所以我们只需要传入 x (取第一个通道 u)
        manu_feat = gen_features_1d(x, length=self.domain_length) # [B, Grid, 10]
        
        # --- C. 特征融合 ---
        # 注意：manu_feat 里面已经包含了 u (index 0) 和 coord (index 9)
        # 所以不需要再 concat 原始 x 了，避免特征冗余
        combined = jnp.concatenate([unet_feat, manu_feat], axis=-1) # Total 74 channels
        
        # --- D. FNO 处理 ---
        x_fno = self.fno_in(combined) 
        x_fno = jax.nn.gelu(x_fno)
        
        for i in range(self.fno_layers):
            x_skip = x_fno
            x_conv = self.cp_conv(x_fno, layer_index=i)
            x_conv = jax.nn.gelu(x_conv)
            x_mix = hk.Linear(x_fno.shape[-1])(x_conv)
            x_mix = jax.nn.gelu(x_mix)
            x_fno = x_skip + x_mix
            x_fno = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x_fno)
                
        # --- E. 输出 ---
        out = self.fno_out_hidden(x_fno)
        out = jax.nn.gelu(out)
        out = self.fno_out(out) # [B, Grid, 1]
        
        return out

# ==============================================================================
# 5. 测试代码
# ==============================================================================


@hk.transform
def conv_fno_1d_forward(x):
    model = FastUNetFNO1D(
        unet_base=16,       
        unet_out=32,        
        fno_width=64,       
        fno_modes=64,       # 之前讨论过，调高到 64
        fno_layers=3,       
        fno_rank=128,       # <---【修改这里】从 64 提升到 128
        domain_length=16.0,
        name="fast_u_net_fno1_d"
    )
    return model(x)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    # 模拟输入: [Batch=10, Grid=200, Channels=2] (u, x)
    input_1d = jax.random.normal(key, (10, 200, 2)) 
    
    print("Initializing Enhanced 1D Burgers Model...")
    params = conv_fno_1d_forward.init(key, input_1d)
    
    # 打印特征提取结果验证一下形状
    dummy_feat = gen_features_1d(input_1d, length=16.0)
    print(f"Manual Features Shape: {dummy_feat.shape} (Should be [10, 200, 10])")
    
    # 跑一次前向传播
    out_1d = conv_fno_1d_forward.apply(params, key, input_1d)
    print(f"Model Output Shape: {out_1d.shape}")