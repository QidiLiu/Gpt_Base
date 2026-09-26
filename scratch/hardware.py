"""硬件与关键环境事实的探针。跑：uv run python scratch/hardware.py"""
import torch
import torch.nn.functional as F
from common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, get_peak_flops, get_peak_bandwidth

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability()
print(f"GPU            : {name}")
print(f"计算能力        : SM{cap[0]}{cap[1]}")
print(f"bf16 支持       : {cap >= (8, 0)}")
print(f"COMPUTE_DTYPE  : {COMPUTE_DTYPE}  ({COMPUTE_DTYPE_REASON})")
print(f"峰值算力        : {get_peak_flops(name)/1e12:.1f} TFLOPS  (fp32 累加口径)")
print(f"峰值带宽        : {get_peak_bandwidth(name)/1e9:.0f} GB/s")
print(f"显存            : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# ─────────────────────────────────────────────────────────────────
# SDPA 的布局陷阱：本项目最危险的 bug 类型
#
# 本项目全程用 (B,T,H,D) 布局（nanochat/FA3 的约定）。
# 但 PyTorch 的 SDPA 严格按 (B,H,L,E) 解读输入。
# 把 (B,T,H,D) 直接喂进去，它会当成 (B,H,T,D) —— 于是
# 「6 个头 x 长度 3」被算成「3 个头 x 长度 6」。
#
# 关键在于：q/k/v 三个张量内部是自洽的，所以
#   · 不报错
#   · 输出形状也刚好一样
#   · 只有**数值**是错的
# 这是最坏的一类 bug：跑通、形状对、loss 还能降，就是训出来的东西不对。
# ─────────────────────────────────────────────────────────────────
print("\n--- SDPA 布局陷阱：形状对但数值错（第 11 章细讲）---")
torch.manual_seed(0)
B, T, H, D = 2, 3, 6, 32
q = torch.randn(B, T, H, D, dtype=torch.float64, device="cuda")
k = torch.randn(B, T, H, D, dtype=torch.float64, device="cuda")
v = torch.randn(B, T, H, D, dtype=torch.float64, device="cuda")

want = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
got = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print(f"输入 (B,T,H,D)   = {tuple(q.shape)}   我们要的是 {H} 个头 x 长度 {T}")
print(f"转置后得到       = {tuple(want.shape)}")
print(f"直接喂得到       = {tuple(got.shape)}")
print(f"形状相同         = {want.shape == got.shape}")
print(f"数值相同         = {torch.allclose(want, got)}")
print(f"最大绝对误差     = {(want - got).abs().max().item():.4f}   <- 完全不同的东西")
print("\n只有当 q 的 T 恰好等于 k 的 H 时才会暴露成 RuntimeError；")
print("更多时候是**静默算错**。src/model/layers.py: sdpa_bt() 负责两次 transpose。")
